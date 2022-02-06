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

        self.test_bar = graph_params['weight_split_range']['test_range'][0]
        self.weight_feature_value = list(self.weight['features'].values())[0]
        self.weight_feature_key = list(self.weight['features'].keys())[0]

        # Nodes connected directly to main node
        self.nodes_direct_with_main = []
        for edge in self.edges:
            source_node = self.edges[edge]['source']
            target_node = self.edges[edge]['target']
            if self.main_node == source_node and self.main_node != target_node:
                self.nodes_direct_with_main.append(
                    target_node) if target_node not in self.nodes_direct_with_main else self.nodes_direct_with_main
            if self.main_node == target_node and self.main_node != source_node:
                self.nodes_direct_with_main.append(
                    source_node) if source_node not in self.nodes_direct_with_main else self.nodes_direct_with_main

        # Nodes not directly connected to main node
        self.nodes_not_direct_with_main = list(
            set(self.nodes) - set(self.nodes_direct_with_main) - set([self.main_node]))

        # Find routes for the nodes that not directly connected to main node to the main node
        self.edge_emb = []
        for node in self.nodes_not_direct_with_main:
            for edge in self.edges:
                if 'self_edge' not in self.edges[edge]:
                    source_node = self.edges[edge]['source']
                    target_node = self.edges[edge]['target']
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
        self.G = nx.Graph()
        self.load_graph_data()
        self.add_nodes()
        self.add_edges()
        self.pass_nodes_info()
        self.pass_nodes_emb()
        
        for node, data in self.graph_data.items():
            if type(data) is dict:
                self.graph_data[node] = self.get_data(node)

        for node in self.graph_data:
            self.graph_data[node]['id'] = self.graph_data[node]['id'].astype(
                int)
            self.graph_data[node] = self.graph_data[node].set_index('id')
            if 'repetition' not in self.graph_data[node].columns:
                self.graph_data[node]['repetition'] = 0

    def load_graph_data(self):
        """ Setup graph structure and add embeddings for nodes """
        print(".. Importing data")
        for node in self.nodes:
            node_features = self.nodes[node]['features']
            node_features_values = list(node_features.values())
            reversed_node_features = {
                node_features[i]: i for i in node_features}

            node_file_name = self.nodes[node]['df']
            node_data = self.input_dfs[node_file_name]
            node_data = node_data[node_data.columns.intersection(
                node_features_values)].drop_duplicates()
            node_data = node_data.rename(
                columns=reversed_node_features).set_index('id')

            node_data['id'] = node_data.index
            node_data['type'] = node

            if 'node_emb' in node_features:
                node_data['node_emb'] = node_data['node_emb'].astype(str)
                node_data['node_emb'] = node_data.apply(
                    lambda x: convert_series_to_array(x['node_emb'], sep=' ',dtype=float), axis=1)

            node_data = node_data.to_dict('index')

            if node in self.emb:
               node_data = self.add_node_emb(node,
                                             self.emb[node]['feature'],
                                             node_data,
                                             min_number_of_words=self.emb[node]['min_number_of_words'],
                                             model=self.emb[node]['model'])
            self.graph_data[node] = node_data

    def add_node_emb(self,
                     node,
                     feature,
                     node_data,
                     min_number_of_words=4,
                     model='XLNetTokenizer'):
        """ Build embeddings for the passed node using a transformer model 

        Parameters
        ----------
        node : str
            the name of the node to add embeddings to
        feature : str
            the name of node feature to calculate the embeddings for
        node_data : dict
            contains data of the node
        min_number_of_words: int
            the minimum length of the sequence to calculate the embedding for
        model : str
            the transormer model (default:'XLNetTokenizer')
        """

        # give different name for the main node embeddings, as it will be passed to other nodes
        if node == self.main_node:
            emb_name = 'emb'
        else:
            emb_name = 'node_emb'

        if model == 'XLNetTokenizer':
            tokenizer = tr.XLNetTokenizer.from_pretrained(
                'xlnet-base-cased')
            model = tr.XLNetModel.from_pretrained('xlnet-base-cased',
                                                  output_hidden_states=True,
                                                  output_attentions=True).to(self.device)
        else:
            print('the selected model is not supported')
            return

        print(f'.... Adding embeddings for {node}:{feature}')
        # TODO: find a faster way to calculate embeddings
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
        return node_data

    def add_nodes(self):
        """ Add nodes to graph """
        print(".. Adding nodes to graph")
        for node in self.graph_data:
            print(f'.... > {node}')
            for key, features in self.graph_data[node].items():
                attrs = ''
                for attr in [str(key)+' = "'+str(value)+'"' for key, value in features.items()]:
                    attrs = attrs + attr + ','
                attrs = attrs[:-1].replace('\n', ' ')
                code = f'self.G.add_node({key}, {attrs})'
                exec(code)

    def add_edges(self):
        """ Connect nodes """
        print('.. Connecting nodes:')
        for edge in self.edges:
            source = self.edges[edge]['source']
            target = self.edges[edge]['target']
            source_id = self.nodes[source]['features']['id']
            target_id = self.nodes[target]['features']['id']

            input_df = self.input_dfs[self.edges[edge]['df']]

            if 'self_edge' in self.edges[edge] and 'parent' in self.nodes[source]:
                target_id = self.nodes[source]['parent']['features']['parent_id']

            edge_type_feature = None
            if 'edge_type_feature' in self.edges[edge]:
                edge_type_feature = self.edges[edge]['edge_type_feature']

            weight = None
            if self.weight_feature_value in input_df.columns.tolist():
                weight = self.weight_feature_value

            fields = [source_id, target_id]
            fields += [weight] if weight is not None else []
            fields += [edge_type_feature] if edge_type_feature is not None else []

            print(f'.... > {edge}: {fields}')

            df = input_df[fields].drop_duplicates()

            if weight and edge_type_feature:
                result = [self.G.add_edge(str(row[0]), str(row[1]),
                                          weight=int(row[2]),
                                          edge_type=f'{source}-{target}_{row[3]}')
                          for row in df[fields].to_numpy()]
            elif weight and not edge_type_feature:
                result = [self.G.add_edge(str(row[0]), str(row[1]),
                                          weight=int(row[2]),
                                          edge_type=f'{source}-{target}')
                          for row in df[fields].to_numpy()]
            elif not weight and edge_type_feature:
                result = [self.G.add_edge(str(row[0]), str(row[1]),
                                          weight=None,
                                          edge_type=f'{source}-{target}_{row[2]}')
                          for row in df[fields].to_numpy()]
            else:
                result = [self.G.add_edge(str(row[0]), str(row[1]),
                                          weight=None,
                                          edge_type=f'{source}-{target}')
                          for row in df[fields].to_numpy()]
            del result

    def pass_nodes_info(self):
        """ Calculate main node info and pass it to other nodes """
        print(f'.. Calculate {self.main_node} info as node attribute')
        main_node_ids = self.get_data(self.main_node)[['id']]
        node_repetition_ids = self.get_data(self.node_to_calculate_repitition)[
            'id'].to_list()
        temp = [self.get_node_repetition(int(row[0]), node_repetition_ids, is_main_node=True) 
                for row in main_node_ids.to_numpy()]
        del temp

        for node in self.nodes:
            if node == self.main_node:
                continue
            print(f'.... > Passing info from {self.main_node} to {node}')
            node_ids = self.get_data(node)[['id']]
            main_node_ids = self.get_data(self.main_node)['id'].to_list()
            temp = [self.get_node_repetition(
                int(row[0]), main_node_ids) for row in node_ids.to_numpy()]
            del temp

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
        connected_nodes = np.array(
            [edge[1] for edge in self.G.edges(str(node_id))], dtype=str)
        link_ids = list(set(connected_nodes) & set(node_repetition_ids))
        if is_main_node:
            self.G.nodes[node_id]['repetition'] = len(link_ids)
        else:
            repetition = 0
            for id in link_ids:
                repetition += self.G.nodes[int(id)]['repetition']
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
        main_node_data['emb'] = main_node_data['emb'].str[1:-1]
        main_node_data['emb'] = main_node_data.apply(
            lambda x: convert_series_to_array(x['emb'], sep=',', dtype=float), axis=1)

        for node in self.nodes:
            if node in [self.main_node] + list(self.nodes_not_direct_with_main):
                continue
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
        node_ids = node_data['id'].astype(int).to_list()
        node_pairs = [list(key) for key, value in nx.get_edge_attributes(self.G, 'weight').items(
        ) if int(value if value is not None else 0) <= self.test_bar and (int(key[0]) in node_ids or int(key[1]) in node_ids)]
        node_pairs = self.get_hash_ids(node_pairs, node_with_emb, node)
        node_pairs = np.array(node_pairs).T
        edge_count = node_pairs.shape[1]
        v = np.ones(edge_count)

        m = normalize(coo_matrix((v, node_pairs),
                                 shape=(len(node_ids),
                                        len(node_with_emb_data))))

        out = m.dot(embeddings)
        node_data['emb'] = list(out)
        self.graph_data[node] = node_data

    def get_hash_ids(self, node_pairs, node1, node2):
        pairs = pd.DataFrame(node_pairs, columns=['id1', 'id2'])
        pairs['id1'] = pairs['id1'].astype('str')
        pairs['id2'] = pairs['id2'].astype('str')
        node_data1 = self.get_data(node1)
        node_data1['hashed_id1'] = node_data1.reset_index().index 
        node_data2 = self.get_data(node2)
        node_data2['hashed_id2'] = node_data2.reset_index().index
        pairs = pairs.merge(node_data1, left_on='id1', right_on='id')[
            ['hashed_id1', 'id1', 'id2']]
        pairs = pairs.merge(node_data2, left_on='id2', right_on='id')[
            ['hashed_id2', 'hashed_id1']]
        return pairs.to_numpy()

def get_edge_list(G):
    """ Returns graph edge list """
    edge_list = defaultdict(  # target_type
            lambda: defaultdict(  # source_type
                lambda: defaultdict(  # relation_type
                    lambda: defaultdict(  # target_id
                        lambda: defaultdict(  # source_id(
                            lambda: int  # weight
                        )))))

    for key, value in nx.get_edge_attributes(G, 'edge_type').items():
        edge_list[G.nodes[int(key[1])]['type']][G.nodes[int(key[0])]
                                                            ['type']][value][int(key[1])][int(key[0])] = G[key[1]][key[0]]['weight']
        edge_list[G.nodes[int(key[0])]['type']][G.nodes[int(key[1])]
                                                            ['type']]['rev_'+str(value)][int(key[0])][int(key[1])] = G[key[1]][key[0]]['weight']

    # Cleans the edge list
    clean_edge_list = {}
    # target_type
    for k1 in edge_list:
        if k1 not in clean_edge_list:
            clean_edge_list[k1] = {}
        # source_type
        for k2 in edge_list[k1]:
            if k2 not in clean_edge_list[k1]:
                clean_edge_list[k1][k2] = {}
            # relation_type
            for k3 in edge_list[k1][k2]:
                if k3 not in clean_edge_list[k1][k2]:
                    clean_edge_list[k1][k2][k3] = {}
                # target_idx
                for e1 in edge_list[k1][k2][k3]:
                    edge_count = len(edge_list[k1][k2][k3][e1])
                    if edge_count == 0:
                        continue
                    clean_edge_list[k1][k2][k3][e1] = {}
                    # source_idx
                    for e2 in edge_list[k1][k2][k3][e1]:
                        clean_edge_list[k1][k2][k3][e1][e2] = edge_list[k1][k2][k3][e1][e2]

    return clean_edge_list

def get_types(G):
    """ Get the type of nodes """
    node_types = list(nx.get_node_attributes( G, 'type').values())
    node_types = list(set(node_types))
    return node_types

def get_meta_graph(G):
    """ Get the types of the edges with related node types """
    edge_types = list(nx.get_edge_attributes(G, 'edge_type').values())
    rev_edge_types = ['rev_' + item for item in list(nx.get_edge_attributes(G, 'edge_type').values())]
    return Counter(edge_types + rev_edge_types)

def get_weights(G):
    weights = {}
    for value in list(nx.get_edge_attributes(G, 'weight').values()):
        weights[value] = True
    return weights

def feature_hgt(layer_data, graph, graph_params):
    feature = {}
    weights = {}
    indxs = {}
    texts = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs = np.array(list(layer_data[_type].keys()))
        tims = np.array(list(layer_data[_type].values()))[:, 1]

        if 'node_emb' in graph.graph_data[_type]:
            feature[_type] = np.array(
                list(graph.graph_data[_type].loc[idxs, 'node_emb']), dtype=np.float)
        else:
            feature[_type] = np.zeros([len(idxs), 400])

        feature[_type] = np.concatenate((feature[_type], list(graph.graph_data[_type].loc[idxs, 'emb']),
                                         np.log10(np.array(list(graph.graph_data[_type].loc[idxs, 'repetition'])).reshape(-1, 1) + 0.01)), axis=1)

        weights[_type] = tims
        indxs[_type] = idxs

        if _type == graph_params['main_node']:
            main_node_feature = graph_params['emb'][graph_params['main_node']]['feature']
            texts = np.array(
                list(graph.graph_data[_type].loc[idxs, main_node_feature]), dtype=np.str)
    return feature, weights, indxs, texts


def sample_subgraph(graph, edge_list, weight_range, graph_params, sampled_depth=2, sampled_number=8, inp=None, feature_extractor=feature_hgt):
    """
        Sample Sub-Graph based on the connection of other nodes with currently sampled nodes
        We maintain budgets for each node type, indexed by <node_id, weight>.
        Currently sampled nodes are stored in layer_data.
        After nodes are sampled, we construct the sampled adjacency matrix.
    """
    layer_data = defaultdict(  # target_type
        lambda: {}  # {target_id: [ser, weight]}
    )
    budget = defaultdict(  # source_type
        lambda: defaultdict(  # source_id
            # [sampled_score, weight]
            lambda: [0., 0]
        ))
    new_layer_adj = defaultdict(  # target_type
        lambda: defaultdict(  # source_type
            lambda: defaultdict(  # relation_type
                lambda: []  # [target_id, source_id]
            )))
    """
        For each node being sampled, we find out all its neighborhood, 
        adding the degree count of these nodes in the budget.
        Note that there exist some nodes that have many neighborhoods
        (such as fields, venues), for those case, we only consider 
    """
    def add_budget(te, target_id, target_weight, layer_data, budget):
        for source_type in te:
            tes = te[source_type]
            for relation_type in tes:
                if relation_type == 'self' or target_id not in tes[relation_type]:
                    continue
                adl = tes[relation_type][target_id]
                if len(adl) < sampled_number:
                    sampled_ids = list(adl.keys())
                else:
                    sampled_ids = np.random.choice(
                        list(adl.keys()), sampled_number, replace=False)
                for source_id in sampled_ids:
                    source_weight = adl[source_id]
                    if source_weight == None:
                        source_weight = target_weight
                    if int(source_weight) > np.max(list(weight_range.keys())) or source_id in layer_data[source_type]:
                        continue
                    budget[source_type][source_id][0] += 1. / len(sampled_ids)
                    budget[source_type][source_id][1] = source_weight

    """First adding the sampled nodes then updating budget"""
    for _type in inp:
        for _id, _weight in inp[_type]:
            layer_data[_type][_id] = [len(layer_data[_type]), _weight]
    for _type in inp:
        te = edge_list[_type]
        for _id, _weight in inp[_type]:
            add_budget(te, _id, _weight, layer_data, budget)
    """
        We recursively expand the sampled graph by sampled_depth.
        Each time we sample a fixed number of nodes for each budget,
        based on the accumulated degree.
    """
    for layer in range(sampled_depth):
        sts = list(budget.keys())
        for source_type in sts:
            te = edge_list[source_type]
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
                add_budget(te, k, budget[source_type]
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
    for target_type in edge_list:
        te = edge_list[target_type]
        tld = layer_data[target_type]
        for source_type in te:
            tes = te[source_type]
            sld = layer_data[source_type]
            for relation_type in tes:
                tesr = tes[relation_type]
                for target_key in tld:
                    if target_key not in tesr:
                        continue
                    target_ser = tld[target_key][0]
                    for source_key in tesr[target_key]:
                        """
                            Check whether each link (target_id, source_id) exist in original adjacency matrix
                        """
                        if source_key in sld:
                            source_ser = sld[source_key][0]
                            edge_list[target_type][source_type][relation_type] += [
                                [target_ser, source_ser]]
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
    types = get_types(graph)
    for t in types:
        node_dict[t] = [node_num, len(node_dict)]
        node_num += len(feature[t])

    for t in types:
        node_feature += list(feature[t])
        node_weight += list(weight[t])
        node_type += [node_dict[t][1] for _ in range(len(feature[t]))]

    edge_dict = {e[2]: i for i, e in enumerate(get_meta_graph(graph))}
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
