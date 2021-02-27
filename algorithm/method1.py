"""
omega = sum(node.out_degree())
local influence = alpha * math.sqrt(k_truss^2+d^2)
更新
"""

from algorithm.K_core.k_truss import k_truss
from timeit import default_timer as timer
import math


def read_to_dict():
    result = dict()
    with open("generation/hep_betweenness_centrality.txt", "r") as f:
        data = f.readlines()
    for data_line in data:
        data_line = data_line[:-1].split(' ')
        key = int(data_line[0])
        value = float(data_line[1])
        result[key] = value
    return result


def LI(G, k, k_trusses, total_weight, node_influence_set):
    LI = dict()
    # 首先全部赋值
    for node in G.nodes:
        d = sum([G[node][v]['weight'] for v in G[node]])
        k_tru = k_trusses[node]

        omega = total_weight[node]
        LI[node] = omega * math.sqrt(k_tru ** 2 + d ** 2)  # d ** 2 + k_tru ** 2   ///// omega *
    S = list()
    #print(sorted(LI.items(), key=lambda x: x[1], reverse=True))
    nis = []  # 选取的节点及其影响的节点
    while len(S) < k:
        node, node_LI = sorted(LI.items(), key=lambda x: x[1], reverse=True)[0]
        print(node, node_LI)
        if node not in nis:
            S.append(node)
            nis.extend(node_influence_set[node])
            LI = discount(G, LI, node_influence_set[node])
        del LI[node]
        # print(i)
        # 更新周围节点
    return S


def discount(G, LI, influence_nodes):
    """
    折扣
    :param G: 图对象
    :param LI: 局部影响力，更新
    :param influence_nodes: 更新这些点的邻居节点的LI值
    :return:
    """
    for node in influence_nodes:
        nbs = list(G.neighbors(node))
        for nb in nbs:
            if nb not in LI.keys():
                LI[nb] = -100
            LI[nb] = LI[nb] - LI[node] / len(nbs)
    return LI


def get_total_probability(G):
    """
    d为平均度
    时间复杂度：O(nd)
    :param G:
    :return:
    """
    total_weight = dict()
    for node in G.nodes:
        d = sum([G[node][v]['weight'] for v in G[node]])
        total_weight[node] = d * 0.01
    return total_weight


if __name__ == "__main__":
    import time

    start = time.time()
    from dataPreprocessing.read_txt_nx import read_Graph

    G = read_Graph("../data/graphdata/hep.txt")
    read_time = time.time()
    print('读取网络时间：', read_time - start)
    k_trusses = k_truss(G)
    total_weight = get_total_probability(G)
    from algorithm.greedy1 import get_node_influence_set

    node_influence_set = get_node_influence_set(G)
    list_IC_random_hep = []
    for k in range(10, 51, 10):
        temp_time = timer()
        S = LI(G, k, k_trusses, total_weight, node_influence_set)
        cal_time = timer() - temp_time
        print('method1算法运行时间：', cal_time)
        from diffusion import spread_run_IC

        average_cover_size = spread_run_IC(G, S, 0.01, 1000)
        print('k=', k, '平均覆盖大小：', average_cover_size)

    # for k in range(1, 51):
    #     S = f(G, k)
    #     cal_time = timer() - temp_time
    #     print('myMethod算法运行时间：', cal_time)
    #     print('k = ', k, '选取节点集为：', S)
    #
    #     from algorithm.Spread.NetworkxSpread import spread_run_IIC
    #
    #     average_cover_size = spread_run_IIC(S, G, 1000)
    #     print('k=', k, '平均覆盖大小：', average_cover_size)
    #
    #     list_IC_random_hep.append({
    #         'k': k,
    #         'run time': cal_time,
    #         'average cover size': average_cover_size,
    #         'S': S
    #     })
    #     temp_time = timer()  # 记录当前时间
    #
    # import pandas as pd
    #
    # df_IC_random_hep = pd.DataFrame(list_IC_random_hep)
    # df_IC_random_hep.to_csv('../data/output/method/IIC_method_hep_Graph.csv')
    # print('文件输出完毕——结束')
