"""
算法：基于独立级联模型的degree heuristic
    获取拥有最大度的前k个节点
来源：Wei Chen et al. Efficient influence maximization in Social Networks
"""
import networkx as nx
from algorithm.priorityQueue import PriorityQueue as PQ  # 优先队列
from timeit import default_timer as timer


def degreeHeuristic(G, k, p=0.01):
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
        degree = sum([G[u][v]['weight'] for v in G[u]])
        # degree = len(G[u])
        d.add_task(u, -degree)
    for i in range(k):
        u, priority = d.pop_item()
        S.append(u)
    return S


def degreeHeuristic2(G, k):
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
        degree = sum([G[u][v]['weight'] for v in G[u]])
        # degree = len(G[u])
        d[u] = degree
    for i in range(k):
        u, degree = max(d.iteritems())
        d.pop(u)
        S.append(u)
    return S


if __name__ == "__main__":
    import time

    start = time.time()
    from algorithm.data_handle.read_Graph_networkx import read_Graph
    G = read_Graph("../../data/graphdata/phy.txt")
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    # 单个测试
    # k = 21
    # temp_time = timer()
    # S = degreeHeuristic(G, k)
    # cal_time = timer() - temp_time
    # print('degreeHeuristic算法运行时间：', cal_time)
    # print('k = ', k, '选取节点集为：', S)
    # from algorithm.Spread.Networkx_spread import spread_run_IC
    # average_cover_size = spread_run_IC(G, S,0.01,1000)
    # print('k=', k, '平均覆盖大小：', average_cover_size)

    I = 1000

    list_IC_hep = []
    temp_time = timer()
    #S = degreeHeuristic(G, 10)
    for k in range(1, 51):
        S = degreeHeuristic(G, k)
        cal_time = timer() - temp_time
        print('degreeHeuristic算法运行时间：', cal_time)
        print('k = ', k, '选取节点集为：', S)

        from algorithm.Spread.Networkx_spread import spread_run_IC

        average_cover_size = spread_run_IC(G,S,0.01, 1000)
        print('k=', k, '平均覆盖大小：', average_cover_size)

        list_IC_hep.append({
            'k': k,
            'run time': cal_time,
            'average cover size': average_cover_size,
            'S': S
        })
        temp_time = timer()  # 记录当前时间

    import pandas as pd

    df_IC_hep = pd.DataFrame(list_IC_hep)
    df_IC_hep.to_csv('../../data/output/degree/IC_degree_phy_Graph.csv')
    print('文件输出完毕——结束')
