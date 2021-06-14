"""
最大核覆盖算法

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



def MaxCoreCover(G, k):
    """
    :param G: networkx图对象
    :param k: 种子集合的大小
    :return:
    """
    start_time = timer()
    node_degree = get_node_degree(G)  # 节点的度
    node_core = get_node_core_number(G,node_degree)
    hd = heapdict()
    for u in G.nodes:
        hd[u] = - node_core[u]
    S, timelapse = [], []
    Visited = dict()  # 节点访问标志
    for node in G.nodes:
        Visited[node] = False
    i = 0
    while i < k:
        u,u_value = hd.popitem()
        if not Visited[u]:
            S.append(u)
            timelapse.append(timer() - start_time)
            Visited[u] = True  # 访问标志
            for v in list(G.neighbors(u)):
                Visited[v] = True  # 访问标志
            i = i + 1
    return (S, timelapse)


if __name__ == "__main__":
    start = time()
    G = read_Graph("../../data/graphdata/phy.txt", directed=False)
    # G = nx.read_edgelist("../../data/graphdata/email.txt", nodetype=int, create_using=nx.Graph)  # 其他数据集使用此方式读取
    # fixed_weight(G)
    read_time = time()
    print('读取网络时间：', read_time - start)
    p = 0.01
    I = 1000
    # p_fixed_with_link(G, p)
    p_fixed(G, p)
    # p_random(G)
    # p_inEdge(G)
    algorithm_output = MaxCoreCover(G, 50)
    list_IC_hep = []
    print("p=", p, ",I=", I, ",data=phy,Graph")
    for k in range(1, 51):
        S = algorithm_output[0][:k]
        cur_spread = IC(G, S, I)
        cal_time = algorithm_output[1][k - 1]
        print('CC算法运行时间：', cal_time)
        print('k = ', k, '选取节点集为：', S)
        print('k=', k, '平均覆盖大小：', cur_spread)
        list_IC_hep.append({
            'k': k,
            'run time': cal_time,
            'average cover size': cur_spread,
            'S': S
        })
    # df_IC_hep = pd.DataFrame(list_IC_hep)
    # df_IC_hep.to_csv('../../data/output/test/IC_degreeCover(p=0.01,I=1000)_email_Graph.csv')
    # print('文件输出完毕——结束')
