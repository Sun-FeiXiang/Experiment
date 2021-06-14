import networkx as nx
from heapdict import heapdict
from timeit import default_timer as timer
from model.ICM_nx import spread_run_IC, IC
import math
from preprocessing.read_txt_nx import read_Graph, avg_degree, avg_degree2
import time
from preprocessing.generation_propagation_probability import fixed_probability, p_random, p_fixed,p_inEdge,p_fixed_with_link,fixed_weight
from preprocessing.generation_node_threshold import random_threshold

def get_node_degree(G):
    """
    获取节点的度（两个节点之间至少有一条边）
    :param G:
    :return:节点的度
    """
    d = dict()
    for u in G.nodes:
        d[u] = sum([G[u][v]['weight'] for v in G[u]])  #
    return d

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


def MCDE(G, k):
    start_time = timer()
    edge_truss_num = get_edge_truss_number(G)  # 边的truss值
    node_degree = get_node_degree(G)  # 节点的度
    node_h = get_node_h(G, node_degree)  # 节点的H指数
    node_E = get_node_E(G, edge_truss_num)  # 节点的信息熵
    mcde = heapdict()
    CO_v = dict()
    for u in G.nodes:
        mcde[u] = - math.sqrt(node_degree[u] ** 2 + node_h[u] ** 2) * node_E[u]
        CO_v[u] = False
    S, timelapse = [], []
    i,j = 0,0
    while i < k:
        u, u_pn = mcde.popitem()
        timelapse.append(timer() - start_time)
        sel = True
        for v in S:
            if Embeddeness(G, u, v) > 0:
                sel = False
        if sel:
            S.append(u)
            timelapse.append(timer() - start_time)
    return (S, timelapse)


def Embeddeness(G, A, B):
    A_neighbors = set(G.neighbors(A))
    B_neighbors = set(G.neighbors(B))
    A_B_intersection = A_neighbors.intersection(B_neighbors)
    # A_B_union = A_neighbors.union(B_neighbors)
    return len(A_B_intersection)


if __name__ == "__main__":
    start = time.time()
    G = read_Graph("../../data/graphdata/hep.txt")  # 针对hep和phy数据集使用该函数读取网络
    # G = nx.read_edgelist("../data/graphdata/facebook_combined.txt",nodetype=int,create_using=nx.Graph) #其他数据集使用此方式读取
    # fixed_weight(G)
    read_time = time.time()
    print('读取网络时间：', read_time - start)
    p = 0.01
    I = 1000
    p_fixed(G, p)
    # p_fixed_with_link(G,p)
    algorithm_output = MCDE(G, 50)
    list_IC_hep = []
    print("p=",p,"R,I=",I,",data=hep,Graph")
    for k in range(1, 51):
        S = algorithm_output[0][:k]
        cur_spread = IC(G, S, I)
        cal_time = algorithm_output[1][k - 1]
        print('method算法运行时间：', cal_time)
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

    # df_IC_hep = pd.DataFrame(list_IC_hep)
    # df_IC_hep.to_csv('../data/output/MCDE/IC_MCDE(p=0.01,I=1000)_hep_Graph.csv')
    # print('文件输出完毕——结束')
