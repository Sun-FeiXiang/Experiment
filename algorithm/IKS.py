"""
Identifying influential spreaders in complex networks based on improved k-shell method
原论文中使用SIR模型
"""
import networkx as nx
from heapdict import heapdict
import math
from time import time
from model.SIR import SIR
from preprocessing.read_txt_nx import read_Graph
from preprocessing.generation_propagation_probability import p_fixed, p_random
from model.ICM_nx import IC


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


def get_node_ei(G, total_degree):
    node_ei = dict()
    for node in G.nodes:
        node_ei[node] = len(G[node]) / total_degree
    return node_ei


def calculate_entropy(G, node_ei):
    node_entropy = heapdict()
    for u in G.nodes:
        cur_node_entropy = 0
        for v in G.neighbors(u):
            cur_node_entropy = cur_node_entropy + node_ei[v] * math.log(node_ei[v], math.e)
        node_entropy[u] = cur_node_entropy
    return node_entropy


# 实现论文中的SIR模拟。。由于边的传播概率未知，结果不太对
def IKS_SIR(G):
    start_time = time()
    node_ei = get_node_ei(G)
    node_entropy = calculate_entropy(G, node_ei)
    nn = nx.number_of_nodes(G)
    k = round(nn * 0.2)
    S = []
    for _ in range(k):
        u, u_value = node_entropy.popitem()
        S.append(u)
    result = SIR(G, S, 100, 0.0266, 0.01)
    subGraph = result[5]
    infect_time = dict()  # 感染时间
    for u in subGraph.nodes:
        t = subGraph.nodes[u]['time']
        if t in infect_time.keys():
            infect_time[t] = infect_time[t] + 1
        else:
            infect_time[t] = 1
    return infect_time


def IKS(G, k):
    start_time = time()
    node_degree = get_node_degree(G)
    total_degree = 0
    for d in node_degree.values():
        total_degree = total_degree + d
    node_ei = get_node_ei(G, total_degree)
    node_entropy = calculate_entropy(G, node_ei)
    S, timelapse = [], []
    for _ in range(k):
        u, u_value = node_entropy.popitem()
        S.append(u)
        timelapse.append(time() - start_time)
    return (S, timelapse)


if __name__ == "__main__":
    start = time()
    G = read_Graph("../data/graphdata/hep.txt")  # 针对hep和phy数据集使用该函数读取网络
    # G = nx.read_edgelist("../data/graphdata/NetHEPT.txt",nodetype=int) #其他数据集使用此方式读取
    read_time = time()
    print('读取网络时间：', read_time - start)
    p = 0.01
    Ep = p_fixed(G, p)
    algorithm_output = IKS(G, 50)
    list_IC_hep = []
    for k in range(1, 51):
        S = algorithm_output[0][:k]
        cur_spread = IC(G, S, 10000)
        cal_time = algorithm_output[1][k - 1]
        print('IKS算法运行时间：', cal_time)
        print('k = ', k, '选取节点集为：', S)
        print('k=', k, '平均覆盖大小：', cur_spread)
        list_IC_hep.append({
            'k': k,
            'run time': cal_time,
            'average cover size': cur_spread,
            'S': S
        })
    import pandas as pd

    df_IC_hep = pd.DataFrame(list_IC_hep)
    df_IC_hep.to_csv('../data/output/IKS/IC_IKS(p=0.01)_hep.csv')
    print('文件输出完毕——结束')
