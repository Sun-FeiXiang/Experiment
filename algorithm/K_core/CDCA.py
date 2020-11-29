import networkx as nx
from copy import deepcopy  # copy graph object
from networkx.algorithms.dominating import dominating_set
from algorithm.distribution.normal_distribution import get_normal_distribution_list
from algorithm.centrality.betweenness_centrality import get_hep_betweenness_centrality
from timeit import default_timer as timer
import math

def get_k_cores_sorted(G, Ep):
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
    #print(k_cores_sorted)
    core_influence = dict()  # 该核层的扩散,被激活的节点数！！！可改
    for k in k_cores_num:
        E = nx.k_core(G, k)
        min_dominating_set = dominating_set(E)
        core_influence[k] = len(min_dominating_set)
    print(core_influence)
    #正态分布
    distribution = get_normal_distribution_list(len(core_influence),sigma=math.sqrt(0.01))
    # print(distribution)

    dis_i = 0
    core_influence_sorted = dict()  # 核影响排序
    for key, value in core_influence.items():
        core_influence_sorted[key] = key + value * distribution[dis_i]
        dis_i = dis_i + 1

    core_influence_sorted = sorted(core_influence_sorted.items(), key=lambda item: item[1], reverse=True)
    # print(core_influence_sorted)
    core_ranking = [one[0] for one in core_influence_sorted]
    # print(core_ranking)
    for core in core_ranking:
        k_cores[core] = k_cores_sorted[core]
    k_cores = sorted(k_cores_sorted.items(), reverse=True)
    return k_cores  # 返回核排名以及每个核对应的节点


def mark_overlay(G, node, CO_v, d=1):
    """
    使用bfs覆盖
    :param G: networkx对象
    :param node: 开始节点
    :param d: 度
    :param CO_v:访问标识
    :return: 无，只需将某个节点设置为访问过即可
    """
    q = []  # 队列
    q.append(node)
    level = 0  # 覆盖第几层
    while len(q) > 0 and level < d:
        v = q.pop(0)  # 弹出第一个节点
        G_adj = G.adj[node]
        for key, value in G_adj.items():
            if not CO_v[key]:
                CO_v[key] = True
                q.append(key)
        level = level + 1  # 访问一层


def CDCA(G, k, Ep):
    k_cores = get_k_cores_sorted(G, Ep)
    CO_v = dict()  # 节点覆盖属性
    for node in G.nodes:
        CO_v[node] = False
    choose_Num = 0  # 选择的节点数
    S = []
    for k_cores_line in k_cores:
        key = k_cores_line[0]
        for k_cores_line_one in k_cores_line[1]:
            node = k_cores_line_one[0]
            node_degree = k_cores_line_one[1]
            if choose_Num == k:
                break
            if not CO_v[node]:
                S.append(node)
                # 标记
                mark_overlay(G, node, CO_v)
                choose_Num = choose_Num + 1
        if choose_Num == k:
            break
    return S


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
        S = CDCA(G, k, Ep)
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
    df_IC_random_hep.to_csv('../../data/output/IC_CDCA_same_CCA_hep.csv')
    print('文件输出完毕——结束')
