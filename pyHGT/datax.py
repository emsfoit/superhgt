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

        self.node_types = graph_params['nodes']
        self.edge_types = graph_params['edges']
        self.weight = graph_params['weight']
        self.emb = graph_params['emb']
        self.main_node = graph_params['main_node']
        self.node_to_calculate_repitition = graph_params['node_to_calculate_repitition']

        self.test_bar = graph_params['weight_split_range']['test_range'][0]
        self.weight_feature_value = list(self.weight['features'].values())[0]
        self.weight_feature_key = list(self.weight['features'].keys())[0]

        # Nodes connected directly to main node
        self.nodes_direct_with_main = []
        for edge_type in self.edge_types:
            source_node = self.edge_types[edge_type]['source']
            target_node = self.edge_types[edge_type]['target']
            if self.main_node == source_node and self.main_node != target_node:
                self.nodes_direct_with_main.append(
                    target_node) if target_node not in self.nodes_direct_with_main else self.nodes_direct_with_main
            if self.main_node == target_node and self.main_node != source_node:
                self.nodes_direct_with_main.append(
                    source_node) if source_node not in self.nodes_direct_with_main else self.nodes_direct_with_main

        # Nodes not directly connected to main node
        self.nodes_not_direct_with_main = list(
            set(self.node_types) - set(self.nodes_direct_with_main) - set([self.main_node]))

        # Find routes for the nodes that not directly connected to main node to the main node
        self.edge_emb = []
        for node in self.nodes_not_direct_with_main:
            for edge_type in self.edge_types:
                if 'self_edge' not in self.edge_types[edge_type]:
                    source_node = self.edge_types[edge_type]['source']
                    target_node = self.edge_types[edge_type]['target']
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


    def load_graph_data(self):
        """ Setup graph structure and add embeddings for nodes """
        print(".. Importing data")
        for node_type in self.node_types:
            node_features = self.node_types[node_type]['features']
            node_features_values = list(node_features.values())
            reversed_node_features = {
                node_features[i]: i for i in node_features}

            node_file_name = self.node_types[node_type]['df']
            node_data = self.input_dfs[node_file_name]
            node_data = node_data[node_data.columns.intersection(
                node_features_values)].drop_duplicates()
            node_data = node_data.rename(
                columns=reversed_node_features).set_index('id')

            node_data['id'] = node_data.index
            node_data['type'] = node_type

            if 'node_emb' in node_features:
                node_data['node_emb'] = node_data['node_emb'].astype(str)
                node_data['node_emb'] = node_data.apply(
                    lambda x: convert_series_to_array(x['node_emb'], sep=' ',dtype=float), axis=1)

            node_data = node_data.to_dict('index')

            if node_type in self.emb:
               node_data = self.add_node_emb(node_type,
                                             self.emb[node_type]['feature'],
                                             node_data,
                                             min_number_of_words=self.emb[node_type]['min_number_of_words'],
                                             model=self.emb[node_type]['model'])
            self.graph_data[node_type] = node_data
        print("HI")

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
        for edge_type in self.edge_types:
            source = self.edge_types[edge_type]['source']
            target = self.edge_types[edge_type]['target']
            source_id = self.node_types[source]['features']['id']
            target_id = self.node_types[target]['features']['id']

            input_df = self.input_dfs[self.edge_types[edge_type]['df']]

            if 'self_edge' in self.edge_types[edge_type] and 'parent' in self.node_types[source]:
                target_id = self.node_types[source]['parent']['features']['parent_id']

            edge_type_feature = None
            if 'edge_type_feature' in self.edge_types[edge_type]:
                edge_type_feature = self.edge_types[edge_type]['edge_type_feature']

            weight = None
            if self.weight_feature_value in input_df.columns.tolist():
                weight = self.weight_feature_value

            fields = [source_id, target_id]
            fields += [weight] if weight is not None else []
            fields += [edge_type_feature] if edge_type_feature is not None else []

            print(f'.... > {edge_type}: {fields}')

            df = input_df[fields].drop_duplicates()
            df['edge_type'] = f'{source}_{target}_' + df[edge_type_feature] if edge_type_feature else f'{source}_{target}'
            df['weight'] = None if not weight else pd.to_numeric(df[weight])
            df = df[[source_id, target_id, 'weight', 'edge_type']]

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

        for node_type in self.node_types:
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

        for node in self.node_types:
            if node in list(self.nodes_not_direct_with_main):
                for edge in self.edge_emb:
                  if edge[0] == node:
                    print(f'.... > to {node} via {edge[1]}')
                    self.get_emb(node, edge[1], self.graph_data[edge[1]])
            elif node not in [self.main_node]:
                print(f'.... > to {node}')
                self.get_emb(node, self.main_node, main_node_data)

        # Pass embeddings to nodes that are not directly connected to main node"""
        for node in self.nodes_not_direct_with_main:
            for edge in self.edge_emb:
                if edge[0] == node:
                    print(f'.... > to {node} via {edge[1]}')
                    self.get_emb(node, edge[1], self.graph_data[edge[1]])

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
                    if self.G.nodes[t]['type'] == node_with_emb and int(att['weight'] if att['weight'] is not None else 0) <= self.test_bar 
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
            feature[_type] = np.zeros([len(idxs), 400])

        feature[_type] = np.concatenate((feature[_type], list(graph.nodes[node]['emb'] for node in idxs),\
                                         np.log10(np.array([graph.nodes[node]['repetition'] for node in idxs]).reshape(-1, 1) + 0.01)), axis=1)

        weights[_type] = tims
        indxs[_type] = idxs

        if _type == graph_params['main_node']:
            main_node_feature = graph_params['emb'][graph_params['main_node']]['feature']
            texts = np.array([graph.nodes[node][main_node_feature] for node in idxs], dtype=np.str)

    return feature, weights, indxs, texts


def sample_subgraph(graph, weight_range, graph_params, sampled_depth=2, sampled_number=8, inp=None, feature_extractor=feature_extractor):
    """
        Sample Sub-Graph based on the connection of other nodes with currently sampled nodes
        We maintain budgets for each node type, indexed by <node_id, weight>.
        Currently sampled nodes are stored in layer_data.
        After nodes are sampled, we construct the sampled adjacency matrix.
    """
    start_time = time.time()
    layer_data = defaultdict(  # target_type
        lambda: {}  # {target_id: [ser, weight]}
    )
    budget = defaultdict(  # source_type
        lambda: defaultdict(  # source_id
            # [sampled_score, weight]
            lambda: [0., 0]
        ))
    """
        For each node being sampled, we find out all its neighborhood, 
        adding the degree count of these nodes in the budget.
        Note that there exist some nodes that have many neighborhoods
        (such as fields, venues), for those case, we only consider 
    """
    def add_budget(graph, target_id, target_weight, layer_data, budget):
        # edges = [ graph.out_edges(target_id, data=True)]
        # cleaned_edges = [(s, t, att) for s, t, att in edges if att['edge_type'] != "self"]
        dict = {}
        for s, t, att in graph.out_edges(target_id, data=True):
            if att['edge_type'] not in dict:
                dict[att['edge_type']] = []
            elm = {'id': t, 'type': graph.nodes[t]['type'], 'weight': att['weight']}
            dict[att['edge_type']].append(elm)

        for relation_type in dict.keys():
            adl = dict[relation_type]
            if len(adl) < sampled_number:
                samples = adl
            else:
                samples = np.random.choice(adl, sampled_number, replace=False)
            for elm in samples:
                source_weight = elm['weight']
                if source_weight == None:
                    source_weight = target_weight
                if int(source_weight) > np.max(list(weight_range.keys())) or elm['id'] in layer_data[elm['type']]:
                    continue
                budget[elm['type']][elm['id']][0] += 1. / len(samples)
                budget[elm['type']][elm['id']][1] = source_weight

    """First adding the sampled nodes then updating budget"""
    for _type in inp:
        for _id, _weight in inp[_type]:
            layer_data[_type][_id] = [len(layer_data[_type]), int(_weight)]
    for _type in inp:
        for _id, _weight in inp[_type]:
            add_budget(graph, _id, int(_weight), layer_data, budget)
    """
        We recursively expand the sampled graph by sampled_depth.
        Each time we sample a fixed number of nodes for each budget,
        based on the accumulated degree.
    """
    for layer in range(sampled_depth):
        sts = list(budget.keys())
        for source_type in sts:
            keys = np.array(list(budget[source_type].keys()))
            if sampled_number > len(keys):
                """Directly sample all the nodes"""
                sampled_ids = np.arange(len(keys))
            else:
                """Sample based on accumulated degree"""
                #print(type(np.array(list(budget[source_type].values()))[:, 0][0]), np.array(list(budget[source_type].values()))[:, 0][0])
                score = np.array(list(budget[source_type].values()))[:, 0].astype(float) ** 2
                score = score / np.sum(score)
                sampled_ids = np.random.choice(
                    len(score), sampled_number, p=score, replace=False)
            sampled_keys = keys[sampled_ids]

            """First adding the sampled nodes then updating budget."""
            for k in sampled_keys:
                layer_data[source_type][k] = [
                    len(layer_data[source_type]), budget[source_type][k][1]]
            for k in sampled_keys:
                add_budget(graph, k, budget[source_type]
                           [k][1], layer_data, budget)
                budget[source_type].pop(k)

    """Prepare feature, weight and adjacency matrix for the sampled graph"""
    feature, weights, indxs, texts = feature_extractor(
        layer_data, graph, graph_params)

    edge_list = defaultdict(  # target_type
        lambda: defaultdict(  # source_type
            lambda: defaultdict(  # relation_type
                lambda: []  # [target_id, source_id]
            )))
    for _type in layer_data:
        for _key in layer_data[_type]:
            _ser = layer_data[_type][_key][0]
            edge_list[_type][_type]['self'] += [[_ser, _ser]]
    """
        Reconstruct sampled adjacency matrix by checking whether each
        link exist in the original graph
    """
    for u,v,e in graph.edges(data=True):
        source_type = graph.nodes[u]['type']
        target_type = graph.nodes[v]['type']
        if u in layer_data[source_type] and v in layer_data[target_type]:
            relation_type = e['edge_type']
            source_ser = layer_data[source_type][u][0]
            target_ser = layer_data[target_type][v][0]
            edge_list[target_type][source_type][relation_type] += [[target_ser, source_ser]]
    end_time = time.time()
    time_elapsed = (end_time - start_time)
    print("done extracting after: ", time_elapsed, "seconds")

    return feature, weights, edge_list, indxs, texts


def to_torch(feature, weight, edge_list, graph):
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
                    edge_weight += [node_weight[tid] - node_weight[sid] + 120]
    node_feature = torch.FloatTensor(node_feature)
    node_type = torch.LongTensor(node_type)
    edge_weight = torch.LongTensor(edge_weight)
    edge_index = torch.LongTensor(edge_index).t()
    edge_type = torch.LongTensor(edge_type)
    return node_feature, node_type, edge_weight, edge_index, edge_type, node_dict, edge_dict
