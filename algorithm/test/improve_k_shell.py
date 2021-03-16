import numpy as np
import networkx as nx


def node_core_number(g):
    """
    :param G:
    :return: 所有节点的核心值
    """
    G = g.copy()
    k_nodes = dict()
    level = 1
    node_degree = get_node_degree(G)
    while len(node_degree):
        while True:
            level_node_list = []
            for item in node_degree.items():  # 返回节点及其度
                if item[1] <= level:
                    level_node_list.append(item[0])
                    # 这里设置了value是从1开始的；
                    k_nodes[item[0]] = level
            G.remove_nodes_from(level_node_list)
            node_degree = get_node_degree(G)
            if not len(node_degree):
                return k_nodes
            #print(sorted(node_degree.items(),key=lambda x: x[1]))
            if min(node_degree.items(), key=lambda x: x[1])[1] > level:
                break

        level = min(node_degree.items(), key=lambda x: x[1])[1]
    return k_nodes


def get_node_degree(G):
    """
    获取节点的度，两个节点之间至少有一条边
    :param G:
    :return:
    """
    d = dict()
    for u in G.nodes:
        d[u] = sum([G[u][v]['weight'] for v in G[u]])
    return d

if __name__ == '__main__':
    from dataPreprocessing.read_txt_nx import read_Graph

    G = read_Graph("../../data/graphdata/hep.txt")

    k_nodes = node_core_number(G.copy())
    G.remove_edges_from(nx.selfloop_edges(G))
    protein_cores = nx.core_number(G)  # 每个顶点的core值

    differ = set(k_nodes.items()) ^ set(protein_cores.items())
    print('differ',differ)
