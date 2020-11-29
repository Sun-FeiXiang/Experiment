import networkx as nx
from copy import deepcopy  # copy graph object
from networkx.algorithms.dominating import dominating_set
from algorithm.distribution.normal_distribution import get_normal_distribution_list
from algorithm.centrality.betweenness_centrality import get_hep_betweenness_centrality
from timeit import default_timer as timer
import math
import random


def get_k_cores_with_betweenness_centrality(G, Ep):
    """
    获得经过排序的k_cores，按照“传播效益”来排序
    :param G:
    :return: “顺序list”及k_cores
    """
    k_cores = {}  # 字典
    highest_kcore = 0  # 记录最高的k-core值
    G.remove_edges_from(nx.selfloop_edges(G))
    protein_cores = nx.core_number(G)  # 每个顶点的core值

    # 计算介数中心性 hep
    hep_betweenness_centrality = get_hep_betweenness_centrality()
    for protein, k_core in protein_cores.items():
        if highest_kcore < k_core:
            highest_kcore = k_core
        if k_core in k_cores:
            k_cores[k_core].append({protein: hep_betweenness_centrality[protein]})
        else:
            k_cores[k_core] = [{protein: hep_betweenness_centrality[protein]}]
    k_cores_num = list(k_cores.keys())
    # print("keys", k_cores_num)
    # 每一层按照介数中心性排序
    k_cores_sorted = {}
    for key, value_dict in k_cores.items():
        k_cores_sorted_line = {}
        for one_value_dict in value_dict:
            k_cores_sorted_line[list(one_value_dict.keys())[0]] = list(one_value_dict.values())[0]
        k_cores_sorted_line = sorted(k_cores_sorted_line.items(), key=lambda item: item[1], reverse=True)
        k_cores_sorted[key] = k_cores_sorted_line
    # print(k_cores_sorted)
    k_cores = sorted(k_cores_sorted.items(), reverse=True)
    return k_cores  # 返回核排名以及每个核对应的节点


def findCCs(G, Ep):
    # 从图G中移除阻塞边，获得传播图
    E = deepcopy(G)
    edge_rem = [e for e in E.edges() if random.random() < (1 - Ep[e]) ** (E[e[0]][e[1]]['weight'])]
    E.remove_edges_from(edge_rem)
    # 初始化 CC
    CCs = dict()  # 每个组件都反映了组件的成员数
    # BFS获得CCs
    for node in E.nodes():
        CCs[node] = bfs(E, node)
    return CCs


def bfs(E, node):
    """
    :param E: 传播图
    :param node: 节点node
    :return: node在E中可到达的节点集
    """
    visited = set()
    import queue
    q = queue.Queue()
    q.put(node)
    res = []
    while not q.empty():
        u = q.get()
        res.append(u)
        adj = list(E.adj[u].keys())
        if len(adj) != 0:
            for v in adj:
                if v not in visited:
                    visited.add(v)
                    q.put(v)
    return res


def method2(G, k, Ep):
    """
    优先选择核大的，相同核的选择介数中心性大的。利用改进的findCCs覆盖。
    :param G:
    :param k:
    :param Ep:
    :return:
    """
    k_cores = get_k_cores_with_betweenness_centrality(G, Ep)
    CO_v = dict()  # 节点覆盖属性
    for node in G.nodes:
        CO_v[node] = False
    choose_Num = 0  # 选择的节点数
    S = []
    for k_cores_line in k_cores:
        key = k_cores_line[0]
        k_cores_sub_graph = nx.k_core(G, key)
        CCs = findCCs(k_cores_sub_graph, Ep)
        for k_cores_line_one in k_cores_line[1]:
            node = k_cores_line_one[0]
            node_degree = k_cores_line_one[1]
            if choose_Num == k:
                break
            if not CO_v[node]:
                S.append(node)
                # 标记
                mark_overlay(CCs[node], CO_v)
                choose_Num = choose_Num + 1
        if choose_Num == k:
            break
    return S


def mark_overlay(cover_list, CO_v):
    for one in cover_list:
        CO_v[one] = True


if __name__ == "__main__":
    import time

    start = time.time()
    from algorithm.graph_data_handle import read_gpickle

    G = read_gpickle("../../data/graphs/hep.gpickle")
    read_time = time.time()
    print('读取网络时间：', read_time - start)
    # 生成固定的传播概率
    from algorithm.generation_propagation_probability import fixed_probability

    Ep = fixed_probability(G, 0.01)

    I = 1000

    list_IC_random_hep = []
    temp_time = timer()
    for k in range(5, 31, 5):
        S = method2(G, k, Ep)
        cal_time = timer() - temp_time
        print('算法运行时间：', cal_time)
        print('选取节点集为：', S)

        from algorithm.IC.IC import avgIC_cover_size

        average_cover_size = avgIC_cover_size(G, S, 0.01, I)
        print('平均覆盖大小：', average_cover_size)

        list_IC_random_hep.append({
            'k': k,
            'run time': cal_time,
            'average cover size': average_cover_size,
            'S': S
        })
        temp_time = timer()  # 记录当前时间

    import pandas as pd

    df_IC_random_hep = pd.DataFrame(list_IC_random_hep)
    df_IC_random_hep.to_csv('../../data/output/IC_method2_hep.csv')
    print('文件输出完毕——结束')

