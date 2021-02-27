"""
算法：
1.计算所有值的k-truss
2.

"""
import random
import networkx as nx
from algorithm.priorityQueue import PriorityQueue as PQ  # 优先队列
from timeit import default_timer as timer
from algorithm.K_core.k_truss import k_truss


def f(G, k):
    """
    在独立级联模型中查找要传播的初始节点集（带优先级队列）
    输入: G -- networkx图对象
    k -- 需要的节点数
    p -- 传播概率
    输出:
    S -- 选择的k个点的集合
    """
    S = []
    d = PQ()
    node_k_truss = k_truss(G)  # 字典，truss：节点
    for truss, node_list in node_k_truss.items():
        node_list_sorted = get_local_influence(G, node_list)  # 按照节点局部影响力排好序的list
        for node in node_list_sorted:
            if len(S) == k:
                break
            S.append(node)
        if len(S) == k:
            break
    return S


def f2(G, k):
    from algorithm.K_core.kCoreDecomposition import kCoreDecomposition
    S = kCoreDecomposition(G, k)
    return S


def get_local_influence(G, node_list):
    """
    获取局部影响力
    时间复杂度：O(k+mn)
    :param G:
    :param node:
    :return:按照节点影响力
    """
    node_inf_dict = dict()
    for node in node_list:
        stand = G.out_degree(node) - G.in_degree(node)  # 坚定系数，影响还是被影响 O(k)
        cur_R_set = get_R_set(G, node)  # O(mn)
        node_inf_dict[node] = stand * len(cur_R_set)
    #print(node_inf_dict)
    node_inf_list = sorted(node_inf_dict.items(), key=lambda A: A[1], reverse=True)  # 按照影响力排序
    result = [node[0] for node in node_inf_list]
    return result
    # node_R_set = dict()  # 保存所需的节点R集


def get_R_set(G, node):
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
            neighbors = G.adj[seed]
            for node in neighbors.keys():
                weight = neighbors[node]['weight']
                # print(node,weight)
                if node not in activity_nodes:
                    if random.random() > weight:
                        activity_nodes.append(node)
                        new_activity_set.append(node)
        activity_set = new_activity_set
    return activity_nodes


if __name__ == "__main__":
    import time

    start = time.time()
    G = nx.read_weighted_edgelist("../data/graphdata/hep.txt", comments='#', nodetype=int, create_using=nx.DiGraph())
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    # 生成固定的传播概率为0.01
    from dataPreprocessing.generation_propagation_probability import weight_probability_fixed

    weight_probability_fixed(G)

    I = 1000

    list_IC_random_hep = []
    temp_time = timer()

    S = f2(G, 10)
    from diffusion import spread_run_IIC
    average_cover_size = spread_run_IIC(S, G, 1000)
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
