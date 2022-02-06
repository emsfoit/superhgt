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
from collections import defaultdict
from scipy.sparse import coo_matrix
from utils.utils import convert_series_to_array, normalize
import os
import dill
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
    gpu : bool
        to use GPU if available (default True)

    Methods
    -------
    load_graph_data()
        Setup graph structure and return a dict of dicts to be used as nodes
    add_node_emb(node_data, feature):
        Adds embeddings for the passed node using a transformers
    add_edge():
        Adds edges between two nodes
    pass_nodes_info():
        Calculate main node info and pass it to other nodes
    pass_nodes_emb():
        Pass main node embeddings to other nodes
    clean_graph_edge_list():
        Cleans the edge list and returns the graph 
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
        gpu : bool
            to use GPU if available (default True)
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_dfs = input_dfs
        if len(self.input_dfs) == 0:
            print("input data is empty, please make sure to pass data in input_dfs")
            return

        self.graph = Graph()
        self.data = defaultdict(lambda: {})

        self.other_nodes = self.nodes.copy()
        del self.other_nodes[self.main_node]

        self.load_graph_data()
        self.add_edges()
        self.pass_nodes_info()
        self.pass_nodes_emb()
        self.clean_graph_edge_list()

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
                    lambda x: convert_series_to_array(x['node_emb']), axis=1)

            node_data = node_data.to_dict('index')

            if node in self.emb:
                node_data = self.add_node_emb(node,
                                              self.emb[node]['feature'],
                                              node_data,
                                              min_number_of_words=self.emb[node]['min_number_of_words'],
                                              model=self.emb[node]['model'])
            self.data[node] = node_data

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

        print(f'.... Adding embeddings for {node}:{feature}')

        # TODO: find a faster way
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

    def add_edges(self):
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

            print(f'.... {edge} ..using.. {fields}')

            df = input_df[fields].drop_duplicates()

            if weight and edge_type_feature:
                result = [self.graph.add_graph_edge(self.data[source][str(row[0])],
                                              self.data[target][str(row[1])],
                                              weight=int(row[2]),
                                              relation_type=f'{source}-{target}_{row[3]}')
                          for row in df[fields].to_numpy()
                          if str(row[1]) in self.data[target] and str(row[0]) in self.data[source]
                          ]
            elif weight and not edge_type_feature:
                result = [self.graph.add_graph_edge(self.data[source][str(row[0])],
                                              self.data[target][str(row[1])],
                                              weight=int(row[2]),
                                              relation_type=f'{source}-{target}')
                          for row in df[fields].to_numpy()
                          if str(row[1]) in self.data[target] and str(row[0]) in self.data[source]
                          ]
            elif not weight and edge_type_feature:
                result = [self.graph.add_graph_edge(self.data[source][str(row[0])],
                                              self.data[target][str(row[1])],
                                              weight=None,
                                              relation_type=f'{source}-{target}_{row[2]}')
                          for row in df[fields].to_numpy()
                          if str(row[1]) in self.data[target] and str(row[0]) in self.data[source]
                          ]
            else:
                result = [self.graph.add_graph_edge(self.data[source][str(row[0])],
                                              self.data[target][str(row[1])],
                                              weight=None,
                                              relation_type=f'{source}-{target}')
                          for row in df[fields].to_numpy()
                          if str(row[1]) in self.data[target] and str(row[0]) in self.data[source]
                        ]
            del result

    def pass_nodes_info(self):
        """ Calculate main node info and pass it to other nodes"""

        print(f'.. Calculate info for node "{self.main_node}"')
        for idx, node in enumerate(self.graph.node_backward[self.main_node]):
            for rel in self.graph.edge_list[self.main_node][self.node_to_calculate_repitition].keys():
                node['repetition'] = len(
                    self.graph.edge_list[self.main_node][self.node_to_calculate_repitition][rel][idx])

        for other_node in self.other_nodes:
            print(f'.. Passing info from "{self.main_node}" to "{other_node}"')
            for idx, node in enumerate(self.graph.node_backward[other_node]):
                repetition = 0
                for rel in self.graph.edge_list[other_node][self.main_node].keys():
                    for main_idx in self.graph.edge_list[other_node][self.main_node][rel][idx]:
                        main_node = self.graph.node_backward[self.main_node][main_idx]
                        repetition += main_node['repetition']
                node['repetition'] = repetition

    def pass_nodes_emb(self):
        """ Pass main node embeddings to other nodes"""

        # pass emb from main node to the nodes that are connected directly to the main node
        print(f'Calculating embeddings for non-{self.main_node} nodes...')
        df = pd.DataFrame(
            self.graph.node_backward[self.main_node]).fillna(0)
        self.graph.node_feature = {self.main_node: df}
        main_node_embeddings = np.array(list(df['emb']))

        for _type in self.graph.node_backward:
            if _type in [self.main_node] + list(self.nodes_not_direct_with_main):
                continue

            df = pd.DataFrame(self.graph.node_backward[_type])
            node_pairs = []
            for _rel in self.graph.edge_list[_type][self.main_node]:
                for target_idx in self.graph.edge_list[_type][self.main_node][_rel]:
                    for source_idx in self.graph.edge_list[_type][self.main_node][_rel][target_idx]:
                        if self.graph.edge_list[_type][self.main_node][_rel][target_idx][source_idx] <= self.test_bar:
                            node_pairs += [[target_idx, source_idx]]
            if len(node_pairs) == 0:
                continue

            node_pairs = np.array(node_pairs).T
            edge_count = node_pairs.shape[1]
            v = np.ones(edge_count)
            m = normalize(coo_matrix((v, node_pairs),
                                     shape=(len(self.graph.node_backward[_type]), len(self.graph.node_backward[self.main_node]))))

            out = m.dot(main_node_embeddings)
            df['emb'] = list(out)
            self.graph.node_feature[_type] = df

        # Calculate embeddings for the nodes that are not directly connected to the main node
        if len(self.nodes_not_direct_with_main) > 0:
            for node in self.nodes_not_direct_with_main:
                for edge in self.edge_emb:
                    if edge[0] == node:
                        indirect_node_embs = np.array(
                            list(self.graph.node_feature[edge[1]]['emb']))
                        df = pd.DataFrame(self.graph.node_backward[node])
                        node_pairs = []
                        for _rel in self.graph.edge_list[node][edge[1]]:
                            for target_idx in self.graph.edge_list[node][edge[1]][_rel]:
                                for source_idx in self.graph.edge_list[node][edge[1]][_rel][target_idx]:
                                    node_pairs += [[target_idx, source_idx]]

                        node_pairs = np.array(node_pairs).T
                        edge_count = node_pairs.shape[1]
                        v = np.ones(edge_count)
                        m = normalize(coo_matrix((v, node_pairs),
                                                 shape=(len(self.graph.node_backward[node]), len(self.graph.node_backward[edge[1]]))))
                        out = m.dot(indirect_node_embs)
                        df['emb'] = list(out)
                        self.graph.node_feature[node] = df

    def clean_graph_edge_list(self):
        """ Cleans the edge list """
        clean_edge_list = {}
        # target_type
        for k1 in self.graph.edge_list:
            if k1 not in clean_edge_list:
                clean_edge_list[k1] = {}
            # source_type
            for k2 in self.graph.edge_list[k1]:
                if k2 not in clean_edge_list[k1]:
                    clean_edge_list[k1][k2] = {}
                # relation_type
                for k3 in self.graph.edge_list[k1][k2]:
                    if k3 not in clean_edge_list[k1][k2]:
                        clean_edge_list[k1][k2][k3] = {}

                    triple_count = 0
                    # target_idx
                    for e1 in self.graph.edge_list[k1][k2][k3]:
                        edge_count = len(self.graph.edge_list[k1][k2][k3][e1])
                        triple_count += edge_count
                        if edge_count == 0:
                            continue
                        clean_edge_list[k1][k2][k3][e1] = {}
                        # source_idx
                        for e2 in self.graph.edge_list[k1][k2][k3][e1]:
                            clean_edge_list[k1][k2][k3][e1][e2] = self.graph.edge_list[k1][k2][k3][e1][e2]
                    print(k1, k2, k3, triple_count)

        self.graph.edge_list = clean_edge_list

        print('Number of nodes:')
        for node_type in self.graph.node_forward:
            print(f'{node_type}: {len(self.graph.node_forward[node_type]):,}')

        del self.graph.node_backward


class Graph():
    def __init__(self):
        super(Graph, self).__init__()
        '''
            node_forward and node_backward are only used when building the data. 
            Afterwards will be transformed into node_feature by DataFrame
            
            node_forward: name -> node_id
            node_backward: node_id -> feature_dict
            node_feature: a DataFrame containing all features
        '''
        self.node_forward = defaultdict(lambda: {})
        self.node_backward = defaultdict(lambda: [])
        self.node_feature = defaultdict(lambda: [])

        '''
            edge_list: index the adjacency matrix (weight) by 
            <target_type, source_type, relation_type, target_id, source_id>
        '''
        self.edge_list = defaultdict(  # target_type
            lambda: defaultdict(  # source_type
                lambda: defaultdict(  # relation_type
                    lambda: defaultdict(  # target_id
                        lambda: defaultdict(  # source_id(
                            lambda: int  # weight
                        )))))
        self.weights = {}

    def add_graph_node(self, node):
        nfl = self.node_forward[node['type']]
        if node['id'] not in nfl:
            self.node_backward[node['type']] += [node]
            ser = len(nfl)
            nfl[node['id']] = ser
            return ser
        return nfl[node['id']]

    def add_graph_edge(self, source_node, target_node, weight=None, relation_type=None, directed=True):
        edge = [self.add_graph_node(source_node), self.add_graph_node(target_node)]
        '''
            Add bi-directional edges with different relation type
        '''
        self.edge_list[target_node['type']][source_node['type']
                                            ][relation_type][edge[1]][edge[0]] = weight
        if directed:
            self.edge_list[source_node['type']][target_node['type']
                                                ]['rev_' + relation_type][edge[0]][edge[1]] = weight
        else:
            self.edge_list[source_node['type']][target_node['type']
                                                ][relation_type][edge[0]][edge[1]] = weight
        self.weights[weight] = True

    def update_node(self, node):
        nbl = self.node_backward[node['type']]
        ser = self.add_graph_node(node)
        for k in node:
            if k not in nbl[ser]:
                nbl[ser][k] = node[k]

    def get_meta_graph(self):
        types = self.get_types()
        metas = []
        for target_type in self.edge_list:
            for source_type in self.edge_list[target_type]:
                for r_type in self.edge_list[target_type][source_type]:
                    metas += [(target_type, source_type, r_type)]
        return metas

    def get_types(self):
        return list(self.node_feature.keys())


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

        if 'node_emb' in graph.node_feature[_type]:
            feature[_type] = np.array(
                list(graph.node_feature[_type].loc[idxs, 'node_emb']), dtype=np.float)
        else:
            feature[_type] = np.zeros([len(idxs), 400])

        feature[_type] = np.concatenate((feature[_type], list(graph.node_feature[_type].loc[idxs, 'emb']),
                                         np.log10(np.array(list(graph.node_feature[_type].loc[idxs, 'repetition'])).reshape(-1, 1) + 0.01)), axis=1)

        weights[_type] = tims
        indxs[_type] = idxs

        if _type == graph_params['main_node']:
            main_node_feature = graph_params['emb'][graph_params['main_node']]['feature']
            texts = np.array(
                list(graph.node_feature[_type].loc[idxs, main_node_feature]), dtype=np.str)
    return feature, weights, indxs, texts


def sample_subgraph(graph, weight_range, graph_params, sampled_depth=2, sampled_number=8, inp=None, feature_extractor=feature_hgt):
    '''
        Sample Sub-Graph based on the connection of other nodes with currently sampled nodes
        We maintain budgets for each node type, indexed by <node_id, weight>.
        Currently sampled nodes are stored in layer_data.
        After nodes are sampled, we construct the sampled adjacency matrix.
    '''
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
    '''
        For each node being sampled, we find out all its neighborhood, 
        adding the degree count of these nodes in the budget.
        Note that there exist some nodes that have many neighborhoods
        (such as fields, venues), for those case, we only consider 
    '''
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
                    if source_weight > np.max(list(weight_range.keys())) or source_id in layer_data[source_type]:
                        continue
                    budget[source_type][source_id][0] += 1. / len(sampled_ids)
                    budget[source_type][source_id][1] = source_weight

    '''
        First adding the sampled nodes then updating budget.
    '''
    for _type in inp:
        for _id, _weight in inp[_type]:
            layer_data[_type][_id] = [len(layer_data[_type]), _weight]
    for _type in inp:
        te = graph.edge_list[_type]
        for _id, _weight in inp[_type]:
            add_budget(te, _id, _weight, layer_data, budget)
    '''
        We recursively expand the sampled graph by sampled_depth.
        Each time we sample a fixed number of nodes for each budget,
        based on the accumulated degree.
    '''
    for layer in range(sampled_depth):
        sts = list(budget.keys())
        for source_type in sts:
            te = graph.edge_list[source_type]
            keys = np.array(list(budget[source_type].keys()))
            if sampled_number > len(keys):
                '''
                    Directly sample all the nodes
                '''
                sampled_ids = np.arange(len(keys))
            else:
                '''
                    Sample based on accumulated degree
                '''
                score = np.array(list(budget[source_type].values()))[:, 0] ** 2
                score = score / np.sum(score)
                sampled_ids = np.random.choice(
                    len(score), sampled_number, p=score, replace=False)
            sampled_keys = keys[sampled_ids]
            '''
                First adding the sampled nodes then updating budget.
            '''
            for k in sampled_keys:
                layer_data[source_type][k] = [
                    len(layer_data[source_type]), budget[source_type][k][1]]
            for k in sampled_keys:
                add_budget(te, k, budget[source_type]
                           [k][1], layer_data, budget)
                budget[source_type].pop(k)
    '''
        Prepare feature, weight and adjacency matrix for the sampled graph
    '''
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
    '''
        Reconstruct sampled adjacency matrix by checking whether each
        link exist in the original graph
    '''
    for target_type in graph.edge_list:
        te = graph.edge_list[target_type]
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
                        '''
                            Check whether each link (target_id, source_id) exist in original adjacency matrix
                        '''
                        if source_key in sld:
                            source_ser = sld[source_key][0]
                            edge_list[target_type][source_type][relation_type] += [
                                [target_ser, source_ser]]
    return feature, weights, edge_list, indxs, texts


def to_torch(feature, weight, edge_list, graph):
    '''
        Transform a sampled sub-graph into pytorch Tensor
        node_dict: {node_type: <node_number, node_type_ID>} node_number is used to trace back the nodes in original graph.
        edge_dict: {edge_type: edge_type_ID}
    '''
    node_dict = {}
    node_feature = []
    node_type = []
    node_weight = []
    edge_index = []
    edge_type = []
    edge_weight = []

    node_num = 0
    types = graph.get_types()
    for t in types:
        node_dict[t] = [node_num, len(node_dict)]
        node_num += len(feature[t])

    for t in types:
        node_feature += list(feature[t])
        node_weight += list(weight[t])
        node_type += [node_dict[t][1] for _ in range(len(feature[t]))]

    edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
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
                    '''
                        Our weight ranges from 1900 - 2020, largest span is 120.
                    '''
                    edge_weight += [node_weight[tid] - node_weight[sid] + 120]
    node_feature = torch.FloatTensor(node_feature)
    node_type = torch.LongTensor(node_type)
    edge_weight = torch.LongTensor(edge_weight)
    edge_index = torch.LongTensor(edge_index).t()
    edge_type = torch.LongTensor(edge_type)
    return node_feature, node_type, edge_weight, edge_index, edge_type, node_dict, edge_dict
class RenameUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "GPT_GNN.data" or module == 'data':
            renamed_module = "pyHGT.data"
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()
