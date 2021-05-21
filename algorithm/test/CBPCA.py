"""
Core-based edge covering algorithm
基于核的边覆盖算法
覆盖系数 c:(0,1)
优先选择pn=sqrt(d**2+k_s**2)大的节点
然后利用k-truss，计算边的truss值
选择pn值大的点，覆盖周围truss值大的边，并标记相应的点，更新周围pn值

"""
import random

from heapdict import heapdict
from timeit import default_timer as timer
from model.ICM_nx import spread_run_IC, IC
import math
import networkx as nx
import sys
from preprocessing.generation_propagation_probability import p_fixed, fixed_weight, p_random, p_inEdge
from time import time
from preprocessing.read_txt_nx import read_Graph
import pandas as pd


def edge_truss_number(G):
    """
    有修改的，求边的truss值
    :param G:
    :return:节点的truss值
    """
    truss_number = dict()
    for edge in G.edges:  # O(m*k^2)
        start = edge[0]
        end = edge[1]
        start_adj = list(G.adj[start].keys())
        end_adj = list(G.adj[end].keys())
        intersection = [i for i in start_adj if i in end_adj]  # O(k^2)
        inter_node_num = 0
        for inter in intersection:
            inter_node_num = inter_node_num + min(G[start][inter]['weight'], G[end][inter]['weight'])
        truss_number[edge] = inter_node_num + 1 + G[start][end]['weight']  # 边的truss值是交集+2+两点之间边的个数-1
        truss_number[edge[1], edge[0]] = truss_number[edge]  # 无向图
    return truss_number


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
            # print(sorted(node_degree.items(),key=lambda x: x[1]))
            if min(node_degree.items(), key=lambda x: x[1])[1] > level:
                break

        level = min(node_degree.items(), key=lambda x: x[1])[1]
    return k_nodes


# def k_core_subGraph(G, k):
#     node_core_num = get_node_core_number(G)
#     g = G.copy()
#     for node, core_num in node_core_num.items():
#         if core_num == k:
#             g.remove_node(node)
#     return g

def edge_cover(G, node, edge_truss_number,l=2):
    """
    层次遍历，优先选择truss值大的边进行覆盖
    :param G:
    :param node:
    :param edge_truss_number:
    :param c:
    :return: 覆盖到的点集
    """
    q = [node]
    cover_list = []
    level = 0
    c = 0.5
    while len(q) > 0 and level < l:
        u = q.pop(0)
        u_neighbors = list(G.neighbors(u))
        if len(u_neighbors) == 0:
            continue
        # c = 1 / len(u_neighbors)  # c默认设置为入度分之一
        cover_num = round(c * len(u_neighbors))  # 覆盖个数等于覆盖概率乘以邻居个数 四舍五入取整
        adj_truss_number = dict()  # 邻边的truss值
        for v in u_neighbors:
            adj_truss_number[u, v] = edge_truss_number[u, v]
        adj_truss_number = sorted(adj_truss_number.items(), key=lambda x: x[1], reverse=True)
        for i in range(cover_num):
            cover_list.append(adj_truss_number[i][0][1])
        q.extend(cover_list)
        level = level + 1
    return cover_list


def get_node_h(G, node_degree):
    """
    计算节点的H指数，时间复杂度O(nd)
    """
    node_h = dict()
    for u in G.nodes:
        neighbors = list(G.neighbors(u))
        neighbors_degree = []
        for v in neighbors:
            neighbors_degree.append(node_degree[v])
        index_list = list(range(1, len(neighbors) + 1))
        neighbors_degree = sorted(neighbors_degree, reverse=True)
        h = 1
        for i, nd in zip(index_list, neighbors_degree):
            if i == nd:
                h = i
                break
        node_h[u] = h
    return node_h


def get_node_E(G, node_core):
    node_E = dict()
    # E_min = sys.maxsize
    for node in G.nodes:
        cur_E = 0
        neighbors = list(G.neighbors(node))
        neighbors_core = dict()  # 邻居在每个核心的个数core:num
        for u in neighbors:
            cur_node_core = node_core[u]  # 获取当前节点的核心值
            if cur_node_core in neighbors_core.keys():
                neighbors_core[cur_node_core] = neighbors_core[cur_node_core] + 1
            else:
                neighbors_core[cur_node_core] = 1
        for core, num in neighbors_core.items():
            p_i = num / len(neighbors)  # 核心值为
            cur_E = cur_E - p_i * math.log2(p_i)
        # if cur_E_i < E_min:
        #     E_min = cur_E_i
        node_E[node] = cur_E
    # E_i_p = dict()  # 归一化后的信息熵
    # for node, node_E_i in E_i.items():
    #     E_i_p[node] = (E_i[node] - E_min) / (math.log2(max_core_num) - E_min)
    return node_E


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


def CBPCA(G, k):
    """
    :param G: networkx图对象
    :param k: 种子集合的大小
    :return:
    """
    start_time = timer()
    edge_truss_num = edge_truss_number(G)  # 边的truss值
    node_degree = get_node_degree(G)  # 节点的度
    node_h = get_node_h(G, node_degree)  # 节点的H指数
    node_core = get_node_core_number(G, node_degree)  # 节点的核心值
    node_E = get_node_E(G, node_core)  # 节点的信息熵
    pn = heapdict()
    for u in G.nodes:
        pn[u] = -math.sqrt(node_degree[u] ** 2 + node_h[u] ** 2) * node_E[u]
    S, timelapse, S_cover = [], [], []
    for _ in range(k):
        u, u_pn = pn.popitem()
        S.append(u)
        timelapse.append(timer() - start_time)
        cur_cover_list = [u]
        cur_cover_list.extend(edge_cover(G, u, edge_truss_num))  # 当前节点覆盖的节点集
        S_cover.extend(cur_cover_list)
        for cover_one in cur_cover_list:  # 弹出这些节点
            if cover_one in pn.keys():
                pn.pop(cover_one)
    return (S, timelapse)


def cal_node_NI(G):
    node_degree = get_node_degree(G)  # 节点的度
    node_h = get_node_h(G, node_degree)  # 节点的H指数
    node_core = get_node_core_number(G, node_degree)  # 节点的核心值
    node_E = get_node_E(G, node_core)  # 节点的信息熵
    ni = heapdict()
    for u in G.nodes:
        ni[u] = -math.sqrt(node_degree[u] ** 2 + node_h[u] ** 2) * node_E[u]
    info = []
    for _ in range(len(G.nodes)):
        u, u_ni = ni.popitem()
        info.append({
            'u': u,
            'NI': -u_ni
        })
    df_IC_hep = pd.DataFrame(info)
    df_IC_hep.to_csv('../../data/output/facebook_NI.csv')
    print('文件输出完毕——结束')


if __name__ == "__main__":
    start = time()
    G = read_Graph("../../data/graphdata/phy.txt")
    # G = nx.read_edgelist("../../data/graphdata/email.txt", nodetype=int,create_using=nx.DiGraph)  # 其他数据集使用此方式读取
    # fixed_weight(G)
    # cal_node_NI(G)
    read_time = time()
    print('读取网络时间：', read_time - start)
    # p_inEdge(G)
    p=0.05
    p_fixed(G,p)
    algorithm_output = CBPCA(G, 50)
    list_IC_hep = []
    print("p=0.05,I=1000,data=phy,Graph")
    for k in range(1, 51):
        S = algorithm_output[0][:k]
        cur_spread = IC(G, S, 1000)
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
        temp_time = timer()  # 记录当前时间
    import pandas as pd

    df_IC_hep = pd.DataFrame(list_IC_hep)
    df_IC_hep.to_csv('../../data/output/CBPCA/IC_CBPCA(p=0.05,l=2,I=1000)_phy.csv')
    print('文件输出完毕——结束')
