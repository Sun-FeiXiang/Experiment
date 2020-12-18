"""
算法：基于独立级联模型的degree heuristic
    获取拥有最大度的前k个节点
来源：Wei Chen et al. Efficient influence maximization in Social Networks
"""
import networkx as nx
from algorithm.priorityQueue import PriorityQueue as PQ  # 优先队列
from timeit import default_timer as timer


def degreeHeuristic(G, k, p=.01):
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
    for u in G:
        degree = G.degree[u]
        # degree = len(G[u])
        d.add_task(u, -degree)
    for i in range(k):
        u, priority = d.pop_item()
        # print(u, priority)
        S.append(u)
    return S


def degreeHeuristic2(G, k, p=.01):
    """
    在独立级联模型中查找要传播的初始节点集（无优先级队列）
    输入: G -- networkx图对象
    k -- 需要的节点数
    p -- 传播概率
    输出:
    S -- 选择的k个点的集合
    """
    S = []
    d = dict()
    for u in G:
        degree = G.degree[u]
        # degree = len(G[u])
        d[u] = degree
    for i in range(k):
        u, degree = max(d.items())
        d.pop(u)
        S.append(u)
    return S


if __name__ == "__main__":
    import time

    start = time.time()
    G = nx.read_weighted_edgelist("../../data/NetHEPT.txt", comments='#', nodetype=int, create_using=nx.Graph())
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    # 生成固定的传播概率为0.01
    from generation.generation_propagation_probability import weight_probability_fixed

    weight_probability_fixed(G, 0.01)

    I = 1000

    list_IC_random_hep = []
    temp_time = timer()
    #S = degreeHeuristic(G, 10)
    for k in range(1, 51):
        S = degreeHeuristic(G, k)
        cal_time = timer() - temp_time
        print('degreeHeuristic算法运行时间：', cal_time)
        print('k = ', k, '选取节点集为：', S)

        from algorithm.Spread.NetworkxSpread import spread_run

        average_cover_size = spread_run(S, G, 1000)
        print('k=', k, '平均覆盖大小：', average_cover_size)

        list_IC_random_hep.append({
            'k': k,
            'run time': cal_time,
            'average cover size': average_cover_size,
            'S': S
        })
        temp_time = timer()  # 记录当前时间

    import pandas as pd

    df_IC_random_hep = pd.DataFrame(list_IC_random_hep)
    df_IC_random_hep.to_csv('../../data/output/degree/IC_degree_NetHEPT_Graph.csv')
    print('文件输出完毕——结束')
