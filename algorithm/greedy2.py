"""
贪心算法2：
1.计算每个节点被哪些节点影响
2.当两个节点影响的个数相同时，优先选择其出邻居被几个节点影响，选择出邻居的出邻居数大的节点并标记它所影响的节点集。
"""
from copy import copy

import networkx as nx
import random
from algorithm.priorityQueue import PriorityQueue as PQ
from timeit import default_timer as timer
import math
from algorithm.Spread.Networkx_spread import runIC


def get_node_influence_node(G, S, R):
    """

    :param G:
    :param S: 种子集合
    :param R: 迭代次数
    :return:
    """
    node_set = set()  # 记录已经出现的节点集合
    node_frequency = dict()  # 记录节点出现的频率
    avg_len = 0
    for i in range(R):
        influence_set = runIC(G,S,0.01)  # 影响的节点集合
        if len(influence_set) != 0:  # 影响的节点不为空集
            avg_len = avg_len + len(influence_set) / R  # 平均影响大小
            for influence_node in influence_set:
                if influence_node not in node_set:
                    node_frequency[influence_node] = 1
                else:
                    node_frequency[influence_node] += 1
                node_set.add(influence_node)
    node_frequency = sorted(node_frequency.items(), key=lambda x: x[1], reverse=True)
    influence_num = math.ceil(avg_len)  # 影响的节点个数，向上取整
    # print('平均影响大小',influence_num)
    # print('influence frequency',node_frequency)
    i = 0
    result = []  # 影响的节点集
    for node_frequency_one in node_frequency:
        if i == influence_num:
            break
        result.append(node_frequency_one[0])
        i = i + 1
    return result


def get_node_influence_set(G):
    """
    获得节点影响的节点集
    1.每个节点运行IC模型R次（默认20）
    2.计算每次运行的结果集中每个元素出现的频次
    3.按照R次平均的次数c，取频率最高的前c个
    :param G:
    :return:
    """
    node_influence_set = dict()
    for node in G.nodes:
        node_influence_set[node] = get_node_influence_node(G, [node], 100)

    return node_influence_set


def get_node_influenced_set(node_influence_set):
    """
    获得图中所有节点被哪些节点影响的集合
    1.get_node_influence_set() 获得节点影响的节点集
    2.“转置”计算出节点被哪些点影响
    :param node_influence_set:
    :return:
    """
    node_influenced_set = dict()
    for key, value in node_influence_set.items():
        # print(key, value)
        for node in value:
            if node in node_influenced_set.keys():
                node_influenced_set[node].append(key)
            else:
                node_influenced_set[node] = [key]

    return node_influenced_set


def get_most_votes_node(node_influence_set, node_influenced_set):
    """
    得票数最多的点
    :param node_influenced_set:
    :param node_influence_set:
    :return:
    """
    # node_influence_set = sorted(node_influence_set.items(),key=lambda x:len(x[1]),reverse=True)

    # 先按照影响个数排序
    node_votes_num = PQ()  #
    for node, influence_nodes in node_influence_set.items():
        votes_num = 0  # 得票数
        for influence_node in influence_nodes:
            votes_num = votes_num + 1 / len(node_influenced_set[influence_node])  # 1/被影响的个数
        node_votes_num.add_task(node, -votes_num)
    max_node, max_votes_num = node_votes_num.pop_item()
    print(max_votes_num)
    return max_node


def update_influence_set(node_influence_set, node):
    """
    更新节点影响集
    :param node_influence_set:
    :param node:
    :return:
    """
    influence_set = [node]  # 影响节点集合
    influence_set.extend(node_influence_set[node])
    for influence_node in influence_set:
        neighbors = G.neighbors(influence_node)
        for neighbor in neighbors:
            if neighbor in node_influence_set.keys():
                node_influence_set[neighbor] = list(
                    set(node_influence_set[neighbor]).difference(set(influence_set)))  # 作差集
    node_influence_set.pop(node)
    return node_influence_set


def greedy(G, k):
    node_influence_set = get_node_influence_set(G)  # 影响的节点集
    node_influenced_set = get_node_influenced_set(node_influence_set)  # 被影响的节点集

    S = []
    for i in range(k):
        u = get_most_votes_node(node_influence_set, node_influenced_set)
        S.append(u)
        node_influence_set = update_influence_set(node_influence_set, u)  # 更新影响的节点集
        node_influenced_set = get_node_influenced_set(node_influence_set)
    return S


if __name__ == "__main__":
    import time
    start = time.time()
    from algorithm.data_handle.read_Graph_networkx import read_Graph
    G = read_Graph("../data/graphdata/hep.txt", directed=True)
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    E = G.copy()
    temp_time = timer()
    k = 50
    S = greedy(G, k)
    cal_time = timer() - temp_time
    print('greedy算法运行时间：', cal_time)
    print('k = ', k, '选取节点集为：', S)
    # S = [62227, 11078, 14642, 36010, 63113, 16164, 33715, 36860, 9082, 16164]
    # S = [6142, 42819, 66135, 66689, 18844, 16164, 30744, 5138, 38112, 40803, 49418, 36860, 63707, 20394, 29595, 57433, 1441, 14906, 23420, 49295, 43226, 41221, 16164, 30160, 23420, 3423, 19660, 48570, 45319, 1441, 57878, 11599, 11850, 1441, 11850, 48570, 12334, 17370, 6975, 51706, 28083, 29595, 11180, 26913, 14642, 2410, 43686, 38614, 11913, 3624]
    from algorithm.Spread.Networkx_spread import spread_run_IC

    average_cover_size = spread_run_IC(E,S,0.01,1000)
    print('k=', k, '平均覆盖大小：', average_cover_size)

    # list_IC_random_hep = []
    # temp_time = timer()
    # for k in range(1, 51):
    #     S = generalGreedy(G, k)
    #     cal_time = timer() - temp_time
    #     print('generalGreedy算法运行时间：', cal_time)
    #     print('k = ', k, '选取节点集为：', S)
    #
    #     from algorithm.Spread.NetworkxSpread import spread_run_IC
    #
    #     average_cover_size = spread_run_IC(S, G, 1000)
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
    # df_IC_random_hep.to_csv('../../data/output/greedy/IC_generalGreedy_NetHEPT.csv')
    # print('文件输出完毕——结束')
    #
    #
    #
