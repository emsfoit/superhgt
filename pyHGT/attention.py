import pandas as pd
import math
from utils.utils import logger
from pyHGT.datax import *

def handle_attention(graph, attention, edges_attentions, edge_index, node_type, node_dict, indxs):
    node_num_name = {val[1]: key for key, val in node_dict.items()}
    edges_nodes_ids = []
    for i, target in enumerate(edge_index[0]):
        source = edge_index[0][i]
        source_node_num = node_type[source]
        node_source_name = node_num_name[source_node_num.item()]
        source_node_id = indxs[node_source_name][source - node_dict[node_source_name][0]]

        target = edge_index[1][i]
        target_node_num = node_type[target]
        node_target_name = node_num_name[target_node_num.item()]
        target_node_id = indxs[node_target_name][target - node_dict[node_target_name][0]]
        edges_nodes_ids += [[source_node_id, target_node_id]]
    for target in set(edge_index[1].tolist()):
        for index in (edge_index[1] == target).nonzero(as_tuple=True)[0]:
            attention_sum = attention[index].sum().item()
            target_id = edges_nodes_ids[index][0]
            source_id = edges_nodes_ids[index][1]
            target_type = graph.nodes[target_id]['type']
            source_type = graph.nodes[source_id]['type']
            edge_type = graph.get_edge_data(target_id, source_id)['edge_type'] if graph.get_edge_data(target_id, source_id) else "self"
            edges_attentions[target_type][source_type][edge_type][target_id][source_id] = attention_sum
    return edges_attentions

def remove_edges(graph, edges_attentions):
    # TODO: Move to config
    relations = [
        {
            'from': {'target': 'paper', 'source': 'field'},
            'take': 0.2
        },
        {
            'from': {'target': 'paper', 'source': 'author'},
            'take': 0.2
        }
    ]
    edges_to_be_removed = []
    for elm in relations:
        edge = edges_attentions[elm['from']['target']][elm['from']['source']]
        edges_info = []
        for rel1 in edge:
            for target_id in edge[rel1]:
                for source_id in edge[rel1][target_id]:
                    # Most of the info here is for debugging purposes only.
                    edge_info = {
                        'target': target_id,
                        'source': source_id,
                        'rel': rel1,
                        'attention': edge[rel1][target_id][source_id]
                    }
                    edges_info.append(edge_info)
        df = pd.DataFrame(edges_info)
        choosen_edges = df.sort_values(by=['attention'])[:math.ceil(len(df)* elm['take'])][['target', 'source']].to_numpy().tolist()
        logger(f"{len(choosen_edges)}/{len(df)} will be removed")
        edges_to_be_removed += choosen_edges
    logger(f"{len(edges_to_be_removed)} total edges will be removed.")
    remove_edges_hgt(graph, edges_to_be_removed)


def add_fake_edges(graph, edges_attentions):
    # TODO: Move to config
    new_relations = [
        {
            'between': ['author', 'field'],
            'through': [{'target': 'paper', 'source': 'field'}, {'target': 'paper', 'source': 'author'}],
            'take': 0.2
        },
        {
            'between': ['author', 'field'],
            'through': [{'target': 'paper', 'source': 'venue'}, {'target': 'paper', 'source': 'author'}],
            'take': 0.2
        }
    ]
    action_edges = []
    for elm in new_relations:
        edges_1 = edges_attentions[elm['through'][0]['target']][elm['through'][0]['source']]
        edges_2 = edges_attentions[elm['through'][1]['target']][elm['through'][1]['source']]
        edges_info = []
        for rel1 in edges_1:
            for id in edges_1[rel1]:
                new_targets_ids = edges_1[rel1][id].keys()
                for rel2 in edges_2:
                    if id in edges_2[rel2]:
                        new_sources_ids = edges_2[rel2][id].keys()
                        for t in new_targets_ids:
                            for s in new_sources_ids:
                                # Most of the info here is for debugging purposes only.
                                id_t = elm['through'][0]['target']+ "-" + elm['through'][0]['source']
                                id_s = elm['through'][1]['target']+ "-" + elm['through'][1]['source']
                                att_sum = edges_1[rel1][id][t] + edges_2[rel2][id][s]
                                new_edge_info = {
                                    'rel1': id_t,
                                    'rel2': id_s,
                                    'left': t,
                                    'main': id,
                                    'right': s,
                                    'main-left-attention': f"{edges_1[rel1][id][t]:.4f}" ,
                                    'main-right-attention': f"{edges_2[rel2][id][s]:.4f}",
                                    'sum-all-attention': f"{att_sum:.4f}" 
                                }
                                edges_info.append(new_edge_info)
        df = pd.DataFrame(edges_info)
        choosen_edges = df.sort_values(by=['sum-all-attention'], ascending=False)[:math.ceil(len(df)* elm['take'])][['left', 'right']].to_numpy().tolist()
        logger(f"{len(choosen_edges)}/{len(df)} will be added for {'-'.join(elm['between'])}")
        action_edges += choosen_edges

    logger(f"{len(action_edges)} total new edges will be added.")
    add_fake_edges_hgt(graph, action_edges)
