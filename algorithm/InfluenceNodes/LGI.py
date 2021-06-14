"""
Ranking influential nodes in complex networks based on local and global structures
"""
from heapdict import heapdict
from timeit import default_timer as timer
from model.ICM_nx import spread_run_IC, IC
import math
import networkx as nx
from preprocessing.generation_propagation_probability import p_fixed, fixed_weight, p_random, p_inEdge, \
    p_fixed_with_link
from time import time
from preprocessing.read_txt_nx import read_Graph, avg_degree
import pandas as pd

def get_node_core_number(g, node_degree):
    """
    修改的，求节点的核心值
    :param G:
    :return: 所有节点的核心值
    """
    G = g.copy()
    k_nodes = dict()
    level = 1
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
        d[u] = sum([G[u][v]['weight'] for v in G[u]])
    return d

def get_node_LC(G):
    node_R = dict()
    for edge in G.edges:
        start = edge[0]
        end = edge[1]
        start_adj = list(G.adj[start].keys())
        end_adj = list(G.adj[end].keys())
        intersection = [i for i in start_adj if i in end_adj]  # O(k^2)
        node_R[start,end] = len(intersection)/len(list(G.neighbors(start)))
        node_R[end,start] = node_R[start, end]
    # print(node_R)
    node_LC = dict()
    for u in G.nodes:
        cur_LC = 0
        u_neighbors = list(G.neighbors(u))
        for v in u_neighbors:
            cur_LC = cur_LC + node_R[(u,v)]
        node_LC[u] = cur_LC
    return node_LC

def get_node_LI(G,node_degree,node_LC):
    node_LI = dict()
    for u in G.nodes:
        u_neighbors = list(G.neighbors(u))
        sum_degree,sum_LC = 0,0
        for v in u_neighbors:
            sum_degree = sum_degree + node_degree[v]
            sum_LC = sum_LC + node_LC[v]
        if sum_LC == 0:
            node_LI[u] = 0.5 * node_degree[u] / sum_degree
        elif sum_degree == 0:
            node_LI[u] = 0.5 * node_LC[u] / sum_LC
        else:
            node_LI[u] = 0.5 * node_degree[u]/sum_degree + 0.5 * node_LC[u]/sum_LC
    return node_LI

def get_node_GI(G,node_core):
    node_GI = dict()
    for v in G.nodes:
        v_neighbors = list(G.neighbors(v))
        sum_neighbors_core= 0
        for u in v_neighbors:
            sum_neighbors_core = sum_neighbors_core + node_core[u]
        node_GI[v] = node_core[v] + sum_neighbors_core
    return node_GI


def LGI(G, k):
    """
    :param G: networkx图对象
    :param k: 种子集合的大小
    :return:
    """
    start_time = timer()
    node_degree = get_node_degree(G)  # 节点的度
    node_core = get_node_core_number(G, node_degree)  # 节点的核心值
    node_LC = get_node_LC(G)
    node_GI = get_node_GI(G,node_core)
    node_LI = get_node_LI(G,node_degree,node_LC)
    node_Q = dict()
    for v in G.nodes:
        node_Q[v] = node_GI[v] * node_LI[v]
    pn = heapdict()
    for v in G.nodes:
        v_neighbors = list(G.neighbors(v))
        cur_LGI = 0
        for u in v_neighbors:
            cur_LGI = cur_LGI + node_Q[u]
        pn[v] = cur_LGI

    S, timelapse = [], []
    i = 0
    while i < k:
        u, u_pn = pn.popitem()
        timelapse.append(timer() - start_time)
        i = i + 1
        S.append(u)
    return (S, timelapse)


if __name__ == "__main__":
    start = time()
    G = read_Graph("../../data/graphdata/hep.txt", directed=False)
    # G = nx.read_edgelist("../../data/graphdata/email.txt", nodetype=int, create_using=nx.Graph)  # 其他数据集使用此方式读取
    # fixed_weight(G)
    read_time = time()
    print('读取网络时间：', read_time - start)
    p = 0.01
    I = 1000
    # p_fixed_with_link(G, p)
    p_fixed(G, p)
    # p_inEdge(G)
    algorithm_output = LGI(G, 50)
    list_IC_hep = []
    print("p=", p, ",I=", I, ",data=email,Graph")
    for k in range(1, 51):
        S = algorithm_output[0][:k]
        cur_spread = IC(G, S, I)
        cal_time = algorithm_output[1][k - 1]
        print('CBPCA算法运行时间：', cal_time)
        print('k = ', k, '选取节点集为：', S)
        print('k=', k, '平均覆盖大小：', cur_spread)
        list_IC_hep.append({
            'k': k,
            'run time': cal_time,
            'average cover size': cur_spread,
            'S': S
        })
    # df_IC_hep = pd.DataFrame(list_IC_hep)
    # df_IC_hep.to_csv('../../data/output/CBPCA/IC_CBPCA1(p=0.02,I=1000)_DBLP_Graph.csv')
    # print('文件输出完毕——结束')
