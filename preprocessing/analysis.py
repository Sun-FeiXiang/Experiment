import networkx as nx

from model.ICM_nx import spread_run_IC, IC
from heapdict import heapdict
from read_txt_nx import read_Graph
from time import time
from preprocessing.generation_propagation_probability import p_fixed,p_random

def node_core_number(g):
    """
    修改的，求节点的核心值
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
            # print(sorted(node_degree.items(),key=lambda x: x[1]))
            if min(node_degree.items(), key=lambda x: x[1])[1] > level:
                break

        level = min(node_degree.items(), key=lambda x: x[1])[1]
    return k_nodes


def get_node_degree(G):
    """
    获取节点的度（两个节点之间至少有一条边）
    :param G:
    :return:节点的度
    """
    d = dict()
    for u in G.nodes:
        # d[u] = sum([G[u][v]['weight'] for v in G[u]])
        d[u] = len(G[u])
    return d


def get_node_h(G):
    """
    计算节点的H指数，时间复杂度O(nd)
    :param G:
    :return:
    """
    node_h = dict()
    for u in G.nodes:
        neighbors = list(G.neighbors(u))
        neighbors_degree = []
        for v in neighbors:
            # neighbors_degree.append(sum([G[v][w]['weight'] for w in G[v]]))
            neighbors_degree.append(len(G[v]))
        index_list = list(range(1, len(neighbors) + 1))
        neighbors_degree = sorted(neighbors_degree, reverse=True)
        h = 1
        for i, nd in zip(index_list, neighbors_degree):
            if i == nd:
                h = i
                break
        node_h[u] = h
    return node_h


def get_node_influence(G):
    """
    获得节点影响力较大的前50个节点
    :param G:
    :return:
    """
    inf = heapdict()
    for node in G.nodes:
        inf[node] = -IC(G, [node], 1000)
    result = dict()
    for _ in range(nx.number_of_nodes(G)):
        u, u_inf = inf.popitem()
        result[u] = -u_inf
    return result


if __name__ == "__main__":
    start = time()
    #G = read_Graph("../data/graphdata/phy.txt")
    G = nx.read_adjlist("../data/graphdata/DBLP.txt",nodetype=int)
    read_time = time()
    print('读取网络时间：', read_time - start)
    p_fixed(G,0.01)
    node_inf = get_node_influence(G)
    node_core = node_core_number(G)
    node_degree = get_node_degree(G)
    node_h = get_node_h(G)

    info = []
    for u, u_inf in node_inf.items():
        u_core = node_core[u]
        u_degree = node_degree[u]
        u_h = node_h[u]
        info.append({
            'u': u,
            'influence': u_inf,
            'degree': u_degree,
            'core': u_core,
            'h': u_h
        })

    import pandas as pd

    df_IC_hep = pd.DataFrame(info)
    df_IC_hep.to_csv('../data/output/DBLP_all_info.csv')
    print('文件输出完毕——结束')
