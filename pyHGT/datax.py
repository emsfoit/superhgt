"""Code summary

This script to be used to build HGT graph based on any graph structured data,
The weight feature should be specified in the .json file in models

Some part of the code is taken from:
https://github.com/acbull/pyHGT/blob/master/OAG/preprocess_OAG.py
https://github.com/acbull/pyHGT/blob/master/pyHGT/data.py
"""

import torch
import pandas as pd
import numpy as np
import transformers as tr
import networkx as nx
from collections import defaultdict, Counter
from scipy.sparse import coo_matrix
from utils.utils import convert_series_to_array, normalize
import time

def find_in_obj(obj, att, value):
    return next((elm for elm in obj if elm[att] == value), None)

class HGTGraph:
    """
    A class to build HGT graph

    ...

    Attributes
    ----------
    graph_params : dict
        the graph parameters used to build the graph
    input_dfs : dataframes {}
        a dict of dataframes of the input files. {file_name1: df1, file_name2: df2, ...}

    Methods
    -------
    load_graph_data()
        Setup graph structure and add embeddings for nodes
    add_node_emb(node, feature, node_data, min_number_of_words, model)
        Build embeddings for the passed node using a transformer model
    add_nodes()
        Add nodes to graph
    add_edges()
        Connect nodes
    pass_nodes_info()
        Calculate main node info and pass it to other nodes
    get_node_repetition(node_id, node_repetition_ids, is_main_node)
        Get node repetition from another node
    get_data(node_type, data_type)
        get data of specific node
    pass_nodes_emb()
        Pass embeddings from the main node to other nodes
    get_emb(node, node_with_emb_data):
        Get embeddings 
    """

    def __init__(self,
                 graph_params,
                 input_dfs):
        """
        Parameters
        ----------
        graph_params : dict
            the graph parameters used to build the graph
        input_dfs : dataframes {}
            a dict of dataframes of the input files. {file_name1: df1, file_name2: df2, ...}
        """

        self.nodes = graph_params['nodes']
        self.edges = graph_params['edges']
        self.weight = graph_params['weight']
        self.emb = graph_params['emb']
        self.main_node = graph_params['main_node']
        self.node_to_calculate_repitition = graph_params['node_to_calculate_repitition']

        # self.test_bar = graph_params['weight_split_range']['test_range'][0]
        # Nodes connected directly to main node
        self.nodes_direct_with_main = []
        for edge in self.edges:
            source_node = edge['source']
            target_node = edge['target']
            if self.main_node == source_node and self.main_node != target_node:
                self.nodes_direct_with_main.append(
                    target_node) if target_node not in self.nodes_direct_with_main else self.nodes_direct_with_main
            if self.main_node == target_node and self.main_node != source_node:
                self.nodes_direct_with_main.append(
                    source_node) if source_node not in self.nodes_direct_with_main else self.nodes_direct_with_main

        # Nodes not directly connected to main node
        self.nodes_not_direct_with_main = list(
            set([node['name'] for node in self.nodes]) - set(self.nodes_direct_with_main) - set([self.main_node]))

        # Find routes for the nodes that not directly connected to main node to the main node
        self.edge_emb = []
        for node in self.nodes_not_direct_with_main:
            for edge in self.edges:
                if 'self_edge' not in edge:
                    source_node = edge['source']
                    target_node = edge['target']
                    if source_node == node and target_node in self.nodes_direct_with_main:
                        self.edge_emb.append([node, target_node]) if [
                            node, target_node] not in self.edge_emb else self.edge_emb
                    elif target_node == node and source_node in self.nodes_direct_with_main:
                        self.edge_emb.append([node, source_node]) if [
                            node, source_node] not in self.edge_emb else self.edge_emb

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.input_dfs = input_dfs
        if len(self.input_dfs) == 0:
            print("input data is empty, please make sure to pass data in input_dfs")
            return

        self.graph_data = defaultdict(lambda: {})
        self.G = nx.DiGraph()
        self.load_graph_data()
        self.add_nodes()
        self.add_edges()
        self.pass_nodes_info()
        self.pass_nodes_emb()

        # define graph meta
        self.G.graph['weights'] = get_weights(self.G)
        self.G.graph['node_types'] = get_types(self.G)
        self.G.graph['meta'] = get_meta_graph(self.G)
        self.G.graph['main_node_embedding_length'] = 7016
        # edge_list = get_edge_list(self.G)
        # self.G.graph['edge_list'] = edge_list
   
    def load_graph_data(self):
        """ Setup graph structure and add embeddings for nodes """
        print(".. Importing data")
        for node in self.nodes:
            node_column_features = { feature['column_name']: feature['feature_name'] for feature in node['features']}

            node_file_name = node['df']
            node_data = self.input_dfs[node_file_name]
            node_data = node_data[node_data.columns.intersection(
                node_column_features.keys())].drop_duplicates()
            node_data = node_data.rename(
                columns=node_column_features).set_index('id')

            node_data['id'] = node_data.index
            node_data['type'] = node['name']
            
            if 'node_emb' in node_column_features.values():
                node_data['node_emb'] = node_data['node_emb'].astype(str)
                node_data['node_emb'] = node_data.apply(
                    lambda x: convert_series_to_array(x['node_emb'], sep=' ',dtype=float), axis=1)

            node_data = node_data.to_dict('index')

            node_emb_config = find_in_obj(self.emb, 'node', node['name'])
            if node_emb_config:
               node_data = self.add_node_emb(node['name'],
                                             node_emb_config['feature'],
                                             node_data,
                                             min_number_of_words=1,
                                             model=node_emb_config['model'])
            self.graph_data[node['name']] = node_data

    def add_node_emb(self,
                     node_type,
                     feature,
                     node_data,
                     min_number_of_words=4,
                     model='XLNetTokenizer'):
        """ Build embeddings for the passed node_type using a transformer model 

        Parameters
        ----------
        node_type : str
            the name of the node_type to add embeddings to
        feature : str
            the name of node_type feature to calculate the embeddings for
        node_data : dict
            contains data of the node_type
        min_number_of_words: int
            the minimum length of the sequence to calculate the embedding for
        model : str
            the transormer model (default:'XLNetTokenizer')
        """

        # give different name for the main node_type embeddings, as it will be passed to other nodes
        if node_type == self.main_node:
            emb_name = 'emb'
        else:
            emb_name = 'node_emb'

        if model == 'XLNetTokenizer':
            tokenizer = tr.XLNetTokenizer.from_pretrained(
                'xlnet-base-cased')
            model = tr.XLNetModel.from_pretrained('xlnet-base-cased',
                                                  output_hidden_states=True,
                                                  output_attentions=True).to(self.device)
            print(f'.... Adding embeddings for {node_type}:{feature}')
            for key, value in node_data.items():
                try:
                    input_ids = torch.tensor(
                        [tokenizer.encode(value[feature])]).to(self.device)[:, :64]
                    if len(input_ids[0]) < min_number_of_words:
                        continue
                    all_hidden_states, all_attentions = model(input_ids)[-2:]
                    rep = (all_hidden_states[-2][0] * all_attentions[-2]
                        [0].mean(dim=0).mean(dim=0).view(-1, 1)).sum(dim=0)
                    value[emb_name] = rep.tolist()
                except Exception as e:
                    print(e)
        else:
            print('Random number will be used as embeding')
            i = 0
            for key, value in node_data.items():
                i = i +1
                value[emb_name] = i
        return node_data

    def add_nodes(self):
        """ Add nodes to graph """
        self.G.add_nodes_from([(f'{features["type"]}_{key}', features) for node in self.graph_data  for key, features in self.graph_data[node].items()])

    def add_edges(self):
        """ Connect nodes """
        print('.. Connecting nodes:')
        for edge in self.edges:
            source = edge['source']
            target = edge['target']
            source_node = find_in_obj(self.nodes, 'name', source)
            target_node =  find_in_obj(self.nodes, 'name', target) 
            source_id = find_in_obj(source_node['features'], 'feature_name', 'id')['column_name']
            target_id = find_in_obj(target_node['features'], 'feature_name', 'id')['column_name']
            input_df = self.input_dfs[edge['df']]

            if 'self_edge' in edge and  'parent' in source_node:
                target_id = find_in_obj(source_node['parent']['features'], 'feature_name', 'parent_id')['column_name']

            edge_type_feature = None
            if 'edge_type_feature' in edge:
                edge_type_feature = edge['edge_type_feature']

            weight = None
            if edge['df'] == self.weight['df']:
                weight = self.weight['feature']

            fields = [source_id, target_id]
            fields += [weight] if weight is not None else []
            fields += [edge_type_feature] if edge_type_feature is not None else []

            print(f".... > {edge['name']}: {fields}")

            df = input_df[fields].drop_duplicates()
            df['edge_type'] = f'{source}_{target}_' + df[edge_type_feature] if edge_type_feature else f'{source}_{target}'
            df['weight'] = None if not weight else pd.to_numeric(df[weight])
            df = df[[source_id, target_id, 'weight', 'edge_type']]
            # extra check: In case target_id or source_id is not defined drop the row (solve reddit comment-comment error)
            df = df.replace('', np.nan).dropna(subset=[source_id, target_id])

            self.G.add_edges_from([
                    (f"{source}_{row[0]}", f"{target}_{row[1]}", {'weight': row[2], 'edge_type': row[3]}) 
                    for row in df.to_numpy()
                    if f"{source}_{row[0]}" in self.G.nodes and f"{target}_{row[1]}" in self.G.nodes  
            ])
            self.G.add_edges_from([
                (f"{target}_{row[1]}", f"{source}_{row[0]}" , {'weight': row[2], 'edge_type': f'rev_{row[3]}'})
                for row in df.to_numpy()
                if f"{source}_{row[0]}" in self.G.nodes and f"{target}_{row[1]}" in self.G.nodes  
            ])
        print("Edges building done")

    def pass_nodes_info(self):
        """ Calculate main node info and pass it to other nodes """
        
        main_node_ids = self.get_data(self.main_node).index.to_list()
        node_repetition_ids = self.get_data(self.node_to_calculate_repitition).index.to_list() if self.main_node != self.node_to_calculate_repitition else main_node_ids
        
        for id in main_node_ids: self.get_node_repetition(id, node_repetition_ids, is_main_node=True)

        for node_type in self.nodes:
            if node_type != self.main_node:
                print(f'.... > Passing info from {self.main_node} to {node_type}')
                node_ids = self.get_data(node_type).index.to_list()
                for id in node_ids: self.get_node_repetition(id, main_node_ids)

    def get_node_repetition(self, node_id, node_repetition_ids, is_main_node=False):
        """ Get node repetition from another node 
        Parameters
        ----------
        node_id : int
            the node id of the source node
        node_repetition_ids : lst
            the node ids of the node to get data from
        is_main_node: bool
            if it is a main node (default: False)
        """
        # TODO where should we use rev and where we should not
        directed_connected_nodes = list(self.G.neighbors(node_id))
        # connected_nodes = list(nx.all_neighbors(self.G, node_id))
        link_ids = list(set(directed_connected_nodes) & set(node_repetition_ids))
        if is_main_node:
            self.G.nodes[node_id]['repetition'] = len(link_ids)
        else:
            repetition = 0
            for id in link_ids:
                repetition += self.G.nodes[id]['repetition']
            self.G.nodes[node_id]['repetition'] = repetition


    def get_data(self, node_type, data_type='dataframe'):
        """ Get data of specific node

        Parameters
        ----------
        node_type : string
            the node type
        data_type: string, optional
            choose betweet ['dataframe', 'dict'], (default is dataframe)

        Returns
        -------
        A dataframe or dict of the selected node type

        """
        df = pd.DataFrame.from_dict(dict(self.G.nodes(data=True)), orient='index')
        df = df[df['type']==node_type].dropna(axis=1, how='all')
        if data_type == 'dict':
            df = df.set_index('id')
            df['id'] = df.index
            dict_data = df.to_dict('index')
            return dict_data
        else:
            return df

    def pass_nodes_emb(self):
        """ Pass embeddings from the main node to other nodes """

        # Pass embeddings to nodes that are directly connected to main node"""
        print(f'.. Passing embeddings from {self.main_node}')
        main_node_data = self.get_data(self.main_node)
        if isinstance(main_node_data['emb'][0], (np.ndarray)):
            main_node_data['emb'] = main_node_data['emb'].str[1:-1]
            main_node_data['emb'] = main_node_data.apply(
                lambda x: convert_series_to_array(x['emb'], sep=',', dtype=float), axis=1)
        else:
            main_node_data['emb'] = main_node_data['emb']

        for node in self.nodes:
            if node['name'] in list(self.nodes_not_direct_with_main):
                for edge in self.edge_emb:
                  if edge[0] == node['name']:
                    print(f".... > to {node['name']} via {edge[1]}")
                    self.get_emb(node['name'], edge[1], self.graph_data[edge[1]])
            elif node['name'] != self.main_node:
                print(f".... > to {node['name']}")
                self.get_emb(node['name'], self.main_node, main_node_data)

        # Pass embeddings to nodes that are not directly connected to main node"""
        for node_name in self.nodes_not_direct_with_main:
            for edge in self.edge_emb:
                if edge[0] == node_name:
                    print(f".... > to {node_name} via {edge[1]}")
                    self.get_emb(node_name, edge[1], self.graph_data[edge[1]])

    def get_emb(self, node, node_with_emb, node_with_emb_data):
        """ Get embeddings 
        Parameters
        ----------
        node : string
            the node name
        node_with_emb_data: string, optional
            the name of the node to get the embeddings from
        """
        embeddings = np.array(list(node_with_emb_data['emb']))
        node_data = self.get_data(node)

        node_with_emb_ids = node_with_emb_data.index.to_list()
        node_ids = node_data.index.to_list()
         # Node pairs should be the edges between node_with_emb and the node
        node_pairs = [ [node_ids.index(s), node_with_emb_ids.index(t)]
                       for s, t, att in self.G.edges(node_ids, data=True)
                    if self.G.nodes[t]['type'] == node_with_emb
                    #  and int(att['weight'] if att['weight'] is not None else 0) <= self.test_bar 
                    ]
        node_pairs = np.array(node_pairs).T
        edge_count = node_pairs.shape[1]
        v = np.ones(edge_count)
        m = normalize(coo_matrix((v, node_pairs),
                                 shape=(len(node_ids), len(node_with_emb_ids))))

        out = m.dot(embeddings)
        node_data['emb'] = list(out)
        self.graph_data[node] = node_data
        new_node_data  = node_data[['emb']].to_dict()['emb']
        nx.set_node_attributes(self.G, new_node_data, 'emb')

def get_types(G):
    """ Get the type of nodes """
    node_types = list(nx.get_node_attributes( G, 'type').values())
    node_types = list(set(node_types))
    return node_types

def get_meta_graph(G):
    """ Get the types of the edges with related node types """
    edge_types = list(nx.get_edge_attributes(G, 'edge_type').values())
    return Counter(edge_types)

def get_weights(G):
    weights = {}
    for value in list(nx.get_edge_attributes(G, 'weight').values()):
        weights[value] = True
    return weights

def get_edge_list(graph):
    edge_list = defaultdict(  # target_type
            lambda: defaultdict(  # source_type
                lambda: defaultdict(  # relation_type
                    lambda: defaultdict(  # target_id
                        lambda: defaultdict(  # source_id(
                            lambda: int  # weight
                        )))))
    for u,v,e in graph.edges(data=True):
        source_type = graph.nodes[u]['type']
        target_type = graph.nodes[v]['type']
        relation_type = e['edge_type']
        weight = e['weight']
        edge_list[source_type][target_type][relation_type][u][v] = int(weight) if weight else None
    return edge_list

def add_fake_edges_hgt(graph, edges):
    graph.add_edges_from([
            (edge[0], edge[1], {'weight':None, 'edge_type': 'fake_edge'})
            for edge in edges
    ])
    graph.add_edges_from([
            (edge[1], edge[0] , {'weight':None, 'edge_type': f'rev_fake_edge'})
            for edge in edges
    ])
    for edge in edges:
        source_type = graph.nodes[edge[0]]['type']
        target_type = graph.nodes[edge[1]]['type']
        graph.graph['edge_list'][source_type][target_type]['fake_edge'][edge[0]][edge[1]] = None
        graph.graph['edge_list'][target_type][source_type]['rev_fake_edge'][edge[1]][edge[0]] = None

def remove_edges_hgt(graph, edges):
    graph.remove_edges_from([(edge[0], edge[1]) for edge in edges])
    graph.remove_edges_from([(edge[1], edge[0]) for edge in edges])
    # Update grpah['edge_list']
    graph.graph['edge_list'] = get_edge_list(graph)


def feature_extractor(layer_data, graph, graph_params):
    feature = {}
    weights = {}
    indxs = {}
    texts = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs = np.array(list(layer_data[_type].keys()))
        tims = np.array(list(layer_data[_type].values()))[:, 1]

        if 'node_emb' in graph.nodes[idxs[0]]:
            feature[_type] = np.array([graph.nodes[node]['node_emb'] for node in idxs], dtype=np.float)
        else:
            # TODO: Change 768 or 400 to node_emb len 
            feature[_type] = np.zeros([len(idxs), 768])

        feature[_type] = np.concatenate((feature[_type], list(graph.nodes[node]['emb'] for node in idxs),\
                                         np.log10(np.array([graph.nodes[node]['repetition'] for node in idxs]).reshape(-1, 1) + 0.01)), axis=1)

        weights[_type] = tims
        indxs[_type] = idxs

        if _type == graph_params['main_node']:
            main_node_feature = find_in_obj(graph_params['emb'], 'node', graph_params['main_node'])['feature']
            texts = np.array([graph.nodes[node][main_node_feature] for node in idxs], dtype=np.str)

    return feature, weights, indxs, texts


def sample_subgraph(graph, weight_range, graph_params, sampled_depth = 2, sampled_number = 8, inp = None, feature_extractor = feature_extractor):
    '''
        Sample Sub-Graph based on the connection of other nodes with currently sampled nodes
        We maintain budgets for each node type, indexed by <node_id, time>.
        Currently sampled nodes are stored in layer_data.
        After nodes are sampled, we construct the sampled adjacancy matrix.
    '''
    layer_data  = defaultdict( #target_type
                        lambda: {} # {target_id: [ser, time]}
                    )
    budget     = defaultdict( #source_type
                                    lambda: defaultdict(  #source_id
                                        lambda: [0., 0] #[sampled_score, time]
                            ))
    '''
        For each node being sampled, we find out all its neighborhood, 
        adding the degree count of these nodes in the budget.
        Note that there exist some nodes that have many neighborhoods
        (such as fields, venues), for those case, we only consider 
    '''
    def add_budget(te, target_id, target_time, layer_data, budget):
        for source_type in te:
            for relation_type in te[source_type]:
                if relation_type == 'self' or target_id not in te[source_type][relation_type]:
                    continue
                if len(te[source_type][relation_type][target_id]) < sampled_number:
                    sampled_ids = list(te[source_type][relation_type][target_id].keys())
                else:
                    sampled_ids = np.random.choice(list(te[source_type][relation_type][target_id].keys()), sampled_number, replace = False)
                for source_id in sampled_ids:
                    source_time = te[source_type][relation_type][target_id][source_id]
                    if source_time == None:
                        source_time = target_time
                    if int(source_time) > np.max(list(weight_range.keys())) or source_id in layer_data[source_type]:
                        continue
                    budget[source_type][source_id][0] += 1. / len(sampled_ids)
                    budget[source_type][source_id][1] = int(source_time)

    '''
        First adding the sampled nodes then updating budget.
    '''
    for _type in inp:
        for _id, _time in inp[_type]:
            layer_data[_type][_id] = [len(layer_data[_type]), _time]
    for _type in inp:
        te = graph.graph['edge_list'][_type]
        for _id, _time in inp[_type]:
            add_budget(te, _id, _time, layer_data, budget)
    '''
        We recursively expand the sampled graph by sampled_depth.
        Each time we sample a fixed number of nodes for each budget,
        based on the accumulated degree.
    '''
    for layer in range(sampled_depth):
        sts = list(budget.keys())
        for source_type in sts:
            te = graph.graph['edge_list'][source_type]
            keys  = np.array(list(budget[source_type].keys()))
            if sampled_number > len(keys):
                '''
                    Directly sample all the nodes
                '''
                sampled_ids = np.arange(len(keys))
            else:
                '''
                    Sample based on accumulated degree
                '''
                score = np.array(list(budget[source_type].values()))[:,0] ** 2
                score = score / np.sum(score)
                sampled_ids = np.random.choice(len(score), sampled_number, p = score, replace = False) 
            sampled_keys = keys[sampled_ids]
            '''
                First adding the sampled nodes then updating budget.
            '''
            for k in sampled_keys:
                layer_data[source_type][k] = [len(layer_data[source_type]), budget[source_type][k][1]]
            for k in sampled_keys:
                add_budget(te, k, budget[source_type][k][1], layer_data, budget)
                budget[source_type].pop(k)   
    '''
        Prepare feature, time and adjacency matrix for the sampled graph
    '''
    feature, times, indxs, texts = feature_extractor(layer_data, graph, graph_params)
            
    edge_list = defaultdict( #target_type
                        lambda: defaultdict(  #source_type
                            lambda: defaultdict(  #relation_type
                                lambda: [] # [target_id, source_id] 
                                    )))
    for _type in layer_data:
        for _key in layer_data[_type]:
            _ser = layer_data[_type][_key][0]
            edge_list[_type][_type]['self'] += [[_ser, _ser]]
    '''
        Reconstruct sampled adjacancy matrix by checking whether each
        link exist in the original graph
    '''
    # layer_data  = target_type  {target_id: [ser(generated index), time]}
    for target_type in graph.graph['edge_list']:
        for source_type in graph.graph['edge_list'][target_type]:
            for relation_type in graph.graph['edge_list'][target_type][source_type]:
                for target_key in layer_data[target_type]:
                    if target_key not in graph.graph['edge_list'][target_type][source_type][relation_type]:
                        continue
                    target_ser = layer_data[target_type][target_key][0]
                    for source_key in graph.graph['edge_list'][target_type][source_type][relation_type][target_key]:
                        '''
                            Check whether each link (target_id, source_id) exist in original adjacancy matrix
                        '''
                        if source_key in layer_data[source_type]:
                            source_ser = layer_data[source_type][source_key][0]
                            edge_list[target_type][source_type][relation_type] += [[target_ser, source_ser]]

    return feature, times, edge_list, indxs, texts


def to_torch(feature, weight, edge_list, graph, include_fake_edges=False):
    """
        Transform a sampled sub-graph into pytorch Tensor
        node_dict: {node_type: <node_number, node_type_ID>} node_number is used to trace back the nodes in original graph.
        edge_dict: {edge_type: edge_type_ID}
    """
    node_dict = {}
    node_feature = []
    node_type = []
    node_weight = []
    edge_index = []
    edge_type = []
    edge_weight = []

    node_num = 0
    types = graph.graph['node_types']
    for t in types:
        node_dict[t] = [node_num, len(node_dict)]
        node_num += len(feature[t])

    for t in types:
        node_feature += list(feature[t])
        node_weight += list(weight[t])
        node_type += [node_dict[t][1] for _ in range(len(feature[t]))]
    edge_dict = {elm: i for i, elm in enumerate(graph.graph['meta']) }
    if include_fake_edges:
        edge_dict['fake_edge'] = len(edge_dict)
        edge_dict['rev_fake_edge'] = len(edge_dict)
    edge_dict['self'] = len(edge_dict)

    for target_type in edge_list:
        for source_type in edge_list[target_type]:
            for relation_type in edge_list[target_type][source_type]:
                for ii, (ti, si) in enumerate(edge_list[target_type][source_type][relation_type]):
                    tid, sid = ti + \
                        node_dict[target_type][0], si + \
                        node_dict[source_type][0]
                    edge_index += [[sid, tid]]
                    edge_type += [edge_dict[relation_type]]
                    # TODO: Change 120 to a dynamic variable
                    """
                        Our weight ranges from 1900 - 2020, largest span is 120.
                    """
                    # TODO: make it dynamic
                    edge_weight += [int(node_weight[tid]) - int(node_weight[sid]) + 120]
    node_feature = torch.FloatTensor(node_feature)
    node_type = torch.LongTensor(node_type)
    edge_weight = torch.LongTensor(edge_weight)
    edge_index = torch.LongTensor(edge_index).t()
    edge_type = torch.LongTensor(edge_type)
    return node_feature, node_type, edge_weight, edge_index, edge_type, node_dict, edge_dict
