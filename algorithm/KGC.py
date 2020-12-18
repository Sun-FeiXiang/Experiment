""""
my method20201206
基于核算法。
1.k-truss
2.贪心算法
使用一个启发因子：f，来划分两个算法步骤
前一部分优先选择k-truss值大的
后一部分使用贪心算法：
"""

import networkx as nx
import math
import random
from timeit import default_timer as timer


def get_k_cores(G):
    k_cores = {}  # 字典
    highest_kcore = 0  # 记录最高的k-core值
    G.remove_edges_from(nx.selfloop_edges(G))
    protein_cores = nx.core_number(G)  # 每个顶点的core值

    for protein, k_core in protein_cores.items():
        if highest_kcore < k_core:
            highest_kcore = k_core
        if k_core in k_cores:
            k_cores[k_core].append({protein: G.out_degree(protein)})
        else:
            k_cores[k_core] = [{protein: G.out_degree(protein)}]

    # 将k_cores的每一行（一层核）按照度排序，度从大到小
    k_cores_sorted = {}
    for key, value_dict in k_cores.items():
        k_cores_sorted_line = {}
        for one_value_dict in value_dict:
            k_cores_sorted_line[list(one_value_dict.keys())[0]] = list(one_value_dict.values())[0]
        k_cores_sorted_line = sorted(k_cores_sorted_line.items(), key=lambda item: item[1], reverse=True)
        k_cores_sorted[key] = k_cores_sorted_line

    # 将k_cores按照key值排序
    k_cores = sorted(k_cores_sorted.items(), reverse=True)
    # print(k_cores)
    return highest_kcore, k_cores


def node_R_IC(G, node):
    """
    生成节点的R集
    :param G:
    :param node:
    :return:
    """
    activity_set = list()
    activity_set.append(node)
    activity_nodes = list()
    activity_nodes.append(node)
    while activity_set:
        new_activity_set = list()
        for seed in activity_set:
            neightbors = G.adj[seed]
            for node in neightbors.keys():
                weight = neightbors[node]['weight']
                # print(node,weight)
                if node not in activity_nodes:
                    if random.random() < weight:
                        activity_nodes.append(node)
                        new_activity_set.append(node)
        activity_set = new_activity_set
    return activity_nodes


def generate_R_IC(G):
    """
    生成图所有节点的R集
    :param G:
    :return:
    """
    RR = dict()
    for node in G.nodes():
        RR[node] = node_R_IC(G, node)
    return RR


def KGC(H, k, f=0.3):
    G = H.copy()
    k1 = math.ceil(f * k)
    k2 = k - math.ceil(f * k)
    S = []
    highest_kcore, k_cores = get_k_cores(G)
    num_k1 = 0
    R = generate_R_IC(G)
    for k_cores_line in k_cores:
        # print(k_cores_line)
        core_num = k_cores_line[0]
        nodes = k_cores_line[1]
        for node_with_degree in nodes:
            node = node_with_degree[0]
            out_degree = node_with_degree[1]
            if num_k1 == k1 or node not in G.nodes:  # 当数目已经够了或者节点不在图中了
                break
            S.append(node)
            G.remove_node(node)
            G.remove_nodes_from(R[node])
            num_k1 = num_k1 + 1
        if num_k1 == k1:
            break

    num_k2 = 0
    # 图变稀疏了，更新R集
    R = generate_R_IC(G)
    R = sorted(R.items(), key=lambda x: len(x[1]), reverse=True)
    # print(R)
    for i in range(k2):
        v = R[0][0]
        R_v = R[0][1]
        S.append(v)
        G.remove_node(v)
        G.remove_nodes_from(R_v)
        R = generate_R_IC(G)
        R = sorted(R.items(), key=lambda x: len(x[1]), reverse=True)

    return S


if __name__ == "__main__":
    import time

    start = time.time()
    G = nx.read_weighted_edgelist("../data/DBLP.txt", comments='#', nodetype=int, create_using=nx.DiGraph())
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    # 生成固定的传播概率
    from generation.generation_propagation_probability import weight_probability_inEdge
    weight_probability_inEdge(G)

    I = 1000
    result = []
    temp_time = timer()
    for k in range(5, 51, 5):
        S = KGC(G, k, f=0.2)
        cal_time = timer() - temp_time
        print('KGC算法运行时间：', cal_time)
        print('选取节点集为：', S)
        from algorithm.Spread.NetworkxSpread import spread_run

        average_cover_size = spread_run(S, G, I)
        print('k =', k, ', f = 0','平均覆盖大小：', average_cover_size)
        result.append({
            'k': k,
            'run time': cal_time,
            'average cover size': average_cover_size,
            'S': S
        })
        temp_time = timer()  # 记录当前时间
    import pandas as pd

    df_result = pd.DataFrame(result)
    df_result.to_csv('../data/output/KGC/IC_KGC_f=0.2_DBLP.csv')
    print('文件输出完毕——结束')
