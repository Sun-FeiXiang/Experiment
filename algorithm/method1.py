"""
omega = sum(node.out_degree())
local influence = alpha * math.sqrt(k_truss^2+d^2)
更新
"""

from algorithm.K_core.k_truss import k_truss
from algorithm.K_core.k_shell import node_core_number
from timeit import default_timer as timer
import networkx as nx
import math
from algorithm.priorityQueue import PriorityQueue as PQ


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


def LI(G, k):
    k_trusses = k_truss(G)
    k_cores = node_core_number(G)
    total_weight = get_total_out_edge_weight(G)
    bcs = read_to_dict()
    LI = PQ()
    # 首先全部赋值
    for node in G.nodes:
        d = G.degree(node)
        k_tru = k_trusses[node]
        k_core = k_cores[node]
        omega = total_weight[node]
        bc = bcs[node]
        LI.add_task(node, - omega * math.sqrt(k_tru ** 2 + d**2 + bc**2))  # d ** 2 + k_tru ** 2   ///// omega *
    S = list()
    from algorithm.greedy1 import get_node_influence_set
    node_influence_set = get_node_influence_set(G)
    nis = [] #选取的节点及其影响的节点
    while len(S) < k:
        node, node_LI = LI.pop_item()
        if node not in nis:
            S.append(node)
            nis.append(node)
            nis.extend(node_influence_set[node])

        # print(i)
        # 更新周围节点
    return S


def get_total_out_edge_weight(G):
    """
    d为平均度
    时间复杂度：O(nd)
    :param G:
    :return:
    """
    total_weight = dict()
    for node in G.nodes:
        out_edges = G.out_edges(node, data='weight')
        w = 0
        for out_edge in out_edges:
            w = w + out_edge[2]
        total_weight[node] = w
    return total_weight


if __name__ == "__main__":
    import time

    start = time.time()
    G = nx.read_weighted_edgelist("../data/graphdata/hep.txt", comments='#', nodetype=int, create_using=nx.DiGraph())
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    # 生成固定的传播概率为0.01
    from generation.generation_propagation_probability import weight_probability_fixed

    weight_probability_fixed(G)

    I = 1000

    list_IC_random_hep = []
    temp_time = timer()

    S = LI(G, 30)
    from algorithm.Spread.Networkx_spread import spread_run_IC

    average_cover_size = spread_run_IC(S, G, 1000)
    print('平均覆盖大小：', average_cover_size)


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
