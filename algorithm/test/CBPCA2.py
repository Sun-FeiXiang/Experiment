"""
Core-based edge covering algorithm
基于核的边覆盖算法
覆盖系数 c:(0,1)
优先选择pn=sqrt(d**2+k_s**2)大的节点
然后利用k-truss，计算边的truss值
选择pn值大的点，覆盖周围truss值大的边，并标记相应的点，更新周围pn值

"""

from heapdict import heapdict
from timeit import default_timer as timer
from model.ICM_nx import spread_run_IC, IC
import math
import networkx as nx
import sys
from preprocessing.read_txt_nx import read_Graph
import numpy as np
from preprocessing.generation_propagation_probability import p_fixed,p_random

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
            neighbors_degree.append(sum([G[v][w]['weight'] for w in G[v]]))
        index_list = list(range(1, len(neighbors) + 1))
        neighbors_degree = sorted(neighbors_degree, reverse=True)
        h = 1
        for i, nd in zip(index_list, neighbors_degree):
            if i == nd:
                h = i
                break
        node_h[u] = h
    return node_h


def get_node_core_number(g):
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


def get_node_entropy(G, node_core):
    """
    获得计算节点信息熵
    :param G:
    :param node_core:
    :return:
    """
    node_entropy = dict()
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        neighbors_core = dict()  # 邻居在每个核心的个数core:num
        for u in neighbors:
            cur_node_core = node_core[u]  # 获取当前节点的核心值
            if cur_node_core in neighbors_core.keys():
                neighbors_core[cur_node_core] = neighbors_core[cur_node_core] + 1
            else:
                neighbors_core[cur_node_core] = 1
        cur_node_entropy = 0
        for core, num in neighbors_core.items():
            p_i = num / len(neighbors)  # 核心值为
            cur_node_entropy = cur_node_entropy + math.log(p_i, math.e)
        node_entropy[node] = -cur_node_entropy
    return node_entropy

def get_node_GI(node_degree, node_entropy):
    avg_degree = np.mean(list(node_degree.values()))
    node_GI = dict()
    for u, u_entropy in node_entropy.items():
        node_GI[u] = u_entropy * avg_degree
    return node_GI


def get_node_LI(node_degree, node_h):
    node_LI = dict()
    for u, u_d in node_degree.items():
        node_LI[u] = math.sqrt(u_d ** 2 + node_h[u] ** 2)
    return node_LI


def normalized(node_attributes):
    min_node_attribute = min(list(node_attributes.values()))
    max_node_attribute = max(list(node_attributes.values()))
    new_node_attributes = dict()
    for u,u_a in node_attributes.items():
        new_node_attributes[u] = (u_a-min_node_attribute)/(max_node_attribute-min_node_attribute)
    return new_node_attributes

def edge_cover(G, node, edge_truss_number, c, l):
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
    while len(q) > 0 and level < l:
        u = q.pop(0)
        u_neighbors = list(G.neighbors(u))
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


def get_node_local_shell(node_core):
    #获得节点所在的核心层
    result_set = set()
    for n,u_core in node_core.items():
        result_set.add(u_core)
    result_list = list(result_set)
    sorted(result_list,reverse=True)
    index = range(0,len(result_list))
    return dict(zip(result_list,index))

def sigmoid(x,a):
    return 1.0 / (1 + np.exp(-a*x))

def CBPCA(G, k, pp, c, l):
    start_time = timer()
    edge_truss_num = edge_truss_number(G)#获取边的truss值
    node_degree = get_node_degree(G)#获取节点度
    node_h = get_node_h(G)#获取节点的h指数
    node_core = get_node_core_number(G)#获取节点的核心值
    node_entropy = get_node_entropy(G, node_core)#获取节点熵

    node_GI = get_node_GI(node_degree,node_entropy)
    node_GI_n = normalized(node_entropy)
    node_LI = get_node_LI(node_degree,node_h)
    node_LI_n = normalized(node_LI)

    node_local_shell = get_node_local_shell(node_core)
    pn = heapdict()
    for u in G.nodes:
        # u_core = node_core[u]#节点核心值
        # l = node_local_shell[u_core] #核心层位置
        # u_lambda = sigmoid(l,pp)
        pn[u] = -(node_LI[u]+node_GI[u])
    S, timelapse = [], []
    S_cover = []
    for _ in range(k):
        u, u_pn = pn.popitem()
        S.append(u)
        timelapse.append(timer() - start_time)
        cur_cover_list = [u]
        cur_cover_list.extend(edge_cover(G, u, edge_truss_num, c, l))  # 当前节点覆盖的节点集
        S_cover.extend(cur_cover_list)
        for cover_one in cur_cover_list:  # 弹出这些节点
            if cover_one in pn.keys():
                pn.pop(cover_one)

    return (S, timelapse)


if __name__ == "__main__":
    import time

    start = time.time()
    G = read_Graph("../../data/graphdata/hep.txt")
    read_time = time.time()
    # G.add_nodes_from(G.nodes, weight=1)#添加weight的默认值
    print('读取网络时间：', read_time - start)
    p = 0.01
    p_fixed(G,p)
    # for pp in range(10):
    #     algorithm_output = CBPCA(G, 10, pp, 0.1, 2)
    #     cur_spread = IC(G, algorithm_output[0], 1000)
    #     print(cur_spread)
    algorithm_output = CBPCA(G, 50, p, 0.1, 2)
    list_IC_hep = []
    for k in range(1, 51):
        S = algorithm_output[0][:k]
        cur_spread = IC(G, S, 1000)
        cal_time = algorithm_output[1][k - 1]
        print('CBECA算法运行时间：', cal_time)
        print('k = ', k, '选取节点集为：', S)
        print('k=', k, '平均覆盖大小：', cur_spread)
        list_IC_hep.append({
            'k': k,
            'run time': cal_time,
            'average cover size': cur_spread,
            'S': S
        })
    # import pandas as pd
    # df_IC_hep = pd.DataFrame(list_IC_hep)
    # df_IC_hep.to_csv('../../data/output/CBPCA/IC_CBPCA(c=0.1,l=2,p=0.02)_facebook_Graph.csv')
    # print('文件输出完毕——结束')
