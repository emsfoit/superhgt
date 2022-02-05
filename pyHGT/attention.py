import datetime
import os
import sys
import time
import glob
from collections import defaultdict
import pandas as pd
import numpy as np


def extract_attention_list(att, edge_type, edge_index_i, edge_index_j):
    """returns a 
    This function to be called in: 
            torch_geometric.nn.conv.MessagePassing.propagate
    Parameters
    ----------
    att : torch 
        contains the attention between the nodes
    edge_type : torch
        the id of the edge type
    edge_index_i : torch
        the id of the source/target node aliened with the att torch
    edge_index_j : torch
        the id of the source/target node aliened with the att torch
    """
    att_list = defaultdict(lambda: defaultdict())
    for i, x in enumerate(att):
        att_list[i]['node1'] = edge_index_j[i].item()
        att_list[i]['node2'] = edge_index_i[i].item()
        att_list[i]['edge_type'] = edge_type[i].item()
        att_list[i]['attention'] = x.cpu().detach().numpy()
        att_list[i]['attention_mean'] = att[i].mean(dim=0).item()
    att_list_df = pd.DataFrame.from_dict(att_list, orient='index')
    curr_time = datetime.datetime.fromtimestamp(
        time.time()).strftime('%Y%m%d_%H%M%S')
    output_file_name = f"{os.path.dirname(sys.modules['__main__'].__file__)}/tests/OAG_attention_list_{curr_time}.tsv"
    print(output_file_name)
    att_list_df.to_csv(output_file_name, index=False, sep="\t")
    del att_list_df


def map_attention_list(epoch, graph, node_dict, edge_dict, indxs):
    """map the attention score to the original nodes
    Parameters
    ----------
    epoch : int
    graph : other(dill) 
    node_dict : dict
        contains mapping of the node ids to their original types
    edge_dict : dict
        contains mapping of the edge ids to their original edges
    indxs : dict
        nodes ids mapping
    """
    attention_files = glob.glob(
        f"{os.path.dirname(sys.modules['__main__'].__file__)}/tests/OAG_attention_list_*")
    for file in attention_files:
        missing_mapping_counter = defaultdict(lambda: defaultdict())
        attention_list = pd.read_csv(file, sep='\t').to_dict('index')
        for key, value in attention_list.items():
            for node_type, indx_list in node_dict.items():
                if value['node1'] >= indx_list[0]:
                    node1_type = node_type
                    node1_start_indx = indx_list[0]
                if value['node2'] >= indx_list[0]:
                    node2_type = node_type
                    node2_start_indx = indx_list[0]
            # to check if index is out of bounds
            test1 = value['node1']-node1_start_indx
            test2 = value['node2']-node2_start_indx
            if (test1 >= indxs[node1_type].size) | (test2 >= indxs[node2_type].size):
                if (test1 >= indxs[node1_type].size):
                    missing_mapping_counter[node1_type][test1] = True
                    #print(f'index {test1} is out of bounds for axis 0 with size {indxs[node1_type].size}, type: {node1_type} {node_dict[node1_type]}')
                if (test2 >= indxs[node2_type].size):
                    #print(f'index {test2} is out of bounds for axis 0 with size {indxs[node2_type].size}, type: {node2_type} {node_dict[node2_type]}')
                    missing_mapping_counter[node2_type][test2] = True
                continue
            node1_org_id = indxs[node1_type][value['node1']-node1_start_indx]
            node2_org_id = indxs[node2_type][value['node2']-node2_start_indx]
            node1_data = graph.node_feature[node1_type].iloc[node1_org_id]
            node2_data = graph.node_feature[node2_type].iloc[node2_org_id]
            for column1, data1 in node1_data.items():
                if 'emb' in column1:
                    continue
                attention_list[key][f'node1_{column1}'] = data1
            for column2, data2 in node2_data.items():
                if 'emb' in column2:
                    continue
                attention_list[key][f'node2_{column2}'] = data2
            edge = [ekey for ekey, evalue in edge_dict.items() if evalue ==
                    value['edge_type']][0]
            attention_list[key]['edge'] = edge
        attention_list_df = pd.DataFrame.from_dict(
            attention_list, orient='index')
        output_file = f"{file.rsplit('/', 1)[0]}/epoch{epoch}_{file.rsplit('/', 1)[1]}.gz"
        attention_list_df.to_csv(output_file, index=False, compression='gzip', sep="\t")
        del attention_list_df, attention_list
        os.remove(file)
        for x in missing_mapping_counter:
            print(
                f'missing mapping {len(missing_mapping_counter[x])} ids in {x}')


def map_attention_list2(epoch, node_dict, edge_dict, indxs):
    """map the attention score to the original nodes
    Parameters
    ----------
    epoch : int
    node_dict : dict
        contains mapping of the node ids to their original types
    edge_dict : dict
        contains mapping of the edge ids to their original edges
    indxs : dict
        nodes ids mapping
    """
    attention_files = glob.glob(
        f"{os.path.dirname(sys.modules['__main__'].__file__)}/tests/OAG_attention_list_*")

    def mapping(node_id1, node_id2, edge_type):
        for node_type, indx_list in node_dict.items():
            if node_id1 >= indx_list[0]:
                node_org_type1 = node_type
                node_start_indx1 = indx_list[0]
            if node_id2 >= indx_list[0]:
                node_org_type2 = node_type
                node_start_indx2 = indx_list[0]
        test1 = node_id1-node_start_indx1
        test2 = node_id2-node_start_indx2
        if (test1 < indxs[node_org_type1].size) & (test2 < indxs[node_org_type2].size):
            node_org_id1 = indxs[node_org_type1][node_id1-node_start_indx1]
            node_org_id2 = indxs[node_org_type2][node_id2-node_start_indx2]
        elif (test1 >= indxs[node_org_type1].size):
            missing_mapping_counter[node_org_type1][test1] = True
            #print(f'index {test1} is out of bounds for axis 0 with size {indxs[node1_type].size}, type: {node1_type} {node_dict[node1_type]}')
        elif (test2 >= indxs[node_org_type2].size):
            #print(f'index {test2} is out of bounds for axis 0 with size {indxs[node2_type].size}, type: {node2_type} {node_dict[node2_type]}')
            missing_mapping_counter[node_org_type2][test2] = True

        """node1_data = graph.node_feature[node1_type].iloc[node1_org_id]
            node2_data = graph.node_feature[node2_type].iloc[node2_org_id]
            for column1, data1 in node1_data.items():
                if 'emb' in column1:
                    continue
                attention_list[key][f'node1_{column1}'] = data1
            for column2, data2 in node2_data.items():
                if 'emb' in column2:
                    continue
                attention_list[key][f'node2_{column2}'] = data2"""

        edge = [ekey for ekey, evalue in edge_dict.items() if evalue ==
                edge_type][0]

        return node_org_id1, node_org_id2, node_org_type1, node_org_type2, edge

    for file in attention_files:
        missing_mapping_counter = defaultdict(lambda: defaultdict())
        attention_list = pd.read_csv(file, sep='\t')
        attention_list[['node1_org_id','node2_org_id', 'node_org_type1', 'node_org_type2', 'edge']] = attention_list.apply(lambda x: mapping(x['node_id1'], x['node_id2'], x['edge_type']), axis=1, result_type="expand")
        output_file = f"{file.rsplit('/', 1)[0]}/epoch{epoch}_{file.rsplit('/', 1)[1]}"
        attention_list.to_csv(output_file, index=False, compression='gzip', sep="\t")
        del attention_list
        os.remove(file)
        for x in missing_mapping_counter:
            print(
                f'missing mapping {len(missing_mapping_counter[x])} ids in {x}')


def get_weak_strong_attention_lists(attention_files_dir, low_value=0.05, high_value=0.3):
    """Return strong and weak attention lists from attention list files based on the passed thresholds
    Parameters
    ----------
    attention_files_dir : string
    low_value : float 
    high_value : float
    """
    attention_files = glob.glob(attention_files_dir)
    lst = []
    for file_name in attention_files:
        df = pd.read_csv(file_name, index_col=None, header=0, compression='gzip', sep="\t")
        lst.append(df)
    attention_list = pd.concat(lst, axis=0, ignore_index=True)[
        ['node1_id', 'node1_type', 'node2_id', 'node2_type', 'edge', 'attention_mean']]
    agg_attention_list = attention_list.groupby(
        ['node1_id', 'node1_type', 'node2_id', 'node2_type', 'edge']).mean().reset_index()
    attention_list_weak = agg_attention_list[agg_attention_list['attention_mean'] <= low_value]
    attention_list_strong = agg_attention_list[agg_attention_list['attention_mean'] >= high_value]
    return attention_list_weak, attention_list_strong


def update_edge_list(edge_list, node_ids_mapping, updating_edges, remove=True):
    """Add/remove edges from the graph edge list
    Parameters
    ----------
    edge_list : defaultdict
        list of the edges between the nodes 
    node_ids_mapping : dict
        list of the ids and thier mapping in the graph (node_forward)
    updating_edges : []
        list of the edges to remove or add
    remove : bool (default: True)
        choose either to add or remove edges
    """
    del_counter = 0
    for type1 in edge_list:
        for type2 in edge_list[type1]:
            df = updating_edges[(updating_edges['node1_type'] == type1) & (
                updating_edges['node2_type'] == type2)]
            node1_map = pd.DataFrame.from_dict(
                node_ids_mapping[type1], orient='index')
            node2_map = pd.DataFrame.from_dict(
                node_ids_mapping[type2], orient='index')
            merge1 = df.merge(node1_map, how='left', left_on='node1_id',
                              right_on=node1_map.index.astype(int))
            merge2 = merge1.merge(
                node2_map, how='left', left_on='node2_id', right_on=node2_map.index.astype(int))
            att_list_cand = merge2.rename(
                columns={"0_x": "node1_idx", "0_y": "node2_idx"})
            for index, row in att_list_cand.iterrows():
                if row['edge'] in edge_list[type1][type2]:
                    if row['node1_idx'] in edge_list[type1][type2][row['edge']]:
                        if row['node2_idx'] in edge_list[type1][type2][row['edge']][row['node1_idx']]:
                            if remove:
                                del edge_list[type1][type2][row['edge']
                                                            ][row['node1_idx']][row['node2_idx']]
                                del_counter += 1
                            else:
                                print("ADD HERE WHAT TO DO FOR ADDING EDGES")
                    if row['node2_idx'] in edge_list[type1][type2][row['edge']]:
                        if row['node1_idx'] in edge_list[type1][type2][row['edge']][row['node2_idx']]:
                            if remove:
                                del edge_list[type1][type2][row['edge']
                                                            ][row['node2_idx']][row['node1_idx']]
                                del_counter += 1
                            else:
                                print("ADD HERE WHAT TO DO FOR ADDING EDGES")
    print(f'{del_counter} deleted edges')
