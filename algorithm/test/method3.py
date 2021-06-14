"""
Core-based edge covering algorithm
基于核的边覆盖算法
覆盖系数 c=p*10
优先选择pn=sqrt(d**2+k_s**2)大的节点
然后利用k-truss，计算边的truss值
选择pn值大的点，覆盖周围truss值大的边(覆盖时采用当前选择种子节点的邻居个数)，并标记相应的点
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


def get_edge_truss_number(G):
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
            if min(node_degree.items(), key=lambda x: x[1])[1] > level:
                break
        level = min(node_degree.items(), key=lambda x: x[1])[1]
    return k_nodes


def path_cover(G, CO_v, node, c):
    """
    层次遍历，优先选择truss值大的边进行覆盖
    :param G:
    :param CO_v:覆盖标识集
    :param node:节点
    :param c: 每层的覆盖率
    :return: 覆盖到的点集
    """
    q = [node]
    node_neighbors_num = len(list(G.neighbors(node)))  # 节点的邻居个数设置为该节点的覆盖大小
    cover_list = []
    while len(q) > 0 and len(cover_list) < node_neighbors_num:
        u = q.pop(0)
        u_neighbors = list(G.neighbors(u))
        if len(u_neighbors) == 0:  # 如果该点没有邻居则继续
            continue
        cover_num = round(c * len(u_neighbors))  # 覆盖个数等于覆盖概率乘以邻居个数 四舍五入取整
        if cover_num == 0:
            cover_num = 1
        u_cover_list = []
        CO_v[u] = True
        i = 0
        while i < cover_num and i < len(u_neighbors):  # 选择未被覆盖的前几个,i不能无限制的加
            v = u_neighbors[i]  # 顺序选取邻居
            if not CO_v[v]:
                CO_v[v] = True
                u_cover_list.append(v)
            i = i + 1
        cover_list.extend(u_cover_list)  # 总覆盖的节点
        q.extend(u_cover_list)
    choose = True  # 是否选择该节点作为种子节点
    if len(cover_list) < node_neighbors_num:
        choose = False  # 覆盖不够，不选为种子节点
    return choose, cover_list


def get_node_h(G, node_degree):
    """
    计算节点的H指数，时间复杂度O(nd)
    :param G:
    :param node_degree:
    :return:
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


def get_node_E(G, edge_truss):
    """
    根据节点核心值获取节点的信息熵
    :param G:
    :param edge_truss:
    :return:节点的信息熵
    """
    node_E = dict()
    for u in G.nodes:
        cur_E = 0
        neighbors = list(G.neighbors(u))
        neighbors_truss = dict()  # 邻居在每个classes中节点的个数T:n
        for v in neighbors:
            cur_node_core = edge_truss[u, v]  # 获取该边的truss
            if cur_node_core in neighbors_truss.keys():
                neighbors_truss[cur_node_core] = neighbors_truss[cur_node_core] + 1
            else:
                neighbors_truss[cur_node_core] = 1
        for core, num in neighbors_truss.items():
            p_i = num / len(neighbors)
            cur_E = cur_E - p_i * math.log2(p_i)
        node_E[u] = cur_E
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


def get_Not_Visited_neighbors(G, node, Visited):
    """
    获取未访问的邻居个数
    :param G:
    :param node:
    :param Visited:
    :return:
    """
    node_neighbors = list(G.neighbors(node))
    result = []
    for u in node_neighbors:
        if not Visited[u]:
            result.append(u)
    return len(result)


def CBPCA(G, k):
    """
    :param G: networkx图对象
    :param k: 种子集合的大小
    :return:
    """
    start_time = timer()
    edge_truss_num = get_edge_truss_number(G)  # 边的truss值
    node_degree = get_node_degree(G)  # 节点的度
    node_h = get_node_h(G, node_degree)  # 节点的H指数
    node_E = get_node_E(G, edge_truss_num)  # 节点的信息熵
    NI = dict()
    for u in G.nodes:
        NI[u] = math.sqrt(node_degree[u] ** 2 + node_h[u] ** 2) * node_E[u]
    S, timelapse = [], []
    Visited = dict()  # 节点访问标志
    for node in G.nodes:
        Visited[node] = False
    i = 0
    NI = sorted(NI.items(), key=lambda x: x[1], reverse=True)  # 按照影响力排名
    while i < k:
        S.append(NI[i][0])
        timelapse.append(timer() - start_time)
        i = i + 1
    return (S, timelapse)


if __name__ == "__main__":
    start = time()
    G = read_Graph("../../data/graphdata/hep.txt", directed=False)
    # G = nx.read_edgelist("../../data/graphdata/email.txt", nodetype=int, create_using=nx.Graph)  # 其他数据集使用此方式读取
    # fixed_weight(G)
    read_time = time()
    print('读取网络时间：', read_time - start)
    p = 0.05
    I = 1000
    # p_fixed_with_link(G, p)
    p_fixed(G, p)
    # p_inEdge(G)
    algorithm_output = CBPCA(G, 50)
    list_IC_hep = []
    print("p=", p, ",I=", I, ",data=hep,Graph")
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
