"""
算法：基于独立级联的简单折扣启发式算法 single discount heuristic[1]

参考：[1] -- Wei Chen et al. Efficient influence maximization in Social Networks
"""
from priorityQueue import PriorityQueue as PQ  # priority queue
from timeit import default_timer as timer
import networkx as nx


def singleDiscount(G, k):
    """
    在独立级联模型中查找要传播的初始节点集（带优先级队列）
    Input: G -- networkx图对象
    k -- 需要的节点数
    Output:
    S -- 选择的k个点的集合
    """
    S = []  # set of activated nodes
    d = PQ()  # degrees
    for u in G:
        degree = sum([G[u][v]['weight'] for v in G[u]])
        d.add_task(u, -degree)
    for i in range(k):
        u, priority = d.pop_item()
        S.append(u)
        for v in G[u]:
            if v not in S:
                [priority, count, task] = d.entry_finder[v]
                d.add_task(v, priority + G[u][v]['weight'])  # discount degree by the weight of the edge
    return S


if __name__ == "__main__":
    import time

    start = time.time()
    from algorithm.data_handle.read_Graph_networkx import read_Graph
    G = read_Graph('../../data/graphdata/phy.txt')
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    list_IC_hep = []
    temp_time = timer()
    for k in range(1, 51):
        S = singleDiscount(G, k)
        cal_time = timer() - temp_time
        print('singleDiscount算法运行时间：', cal_time)
        print('k = ', k, '选取节点集为：', S)

        from algorithm.Spread.Networkx_spread import spread_run_IC

        average_cover_size = spread_run_IC(G,S,0.01,1000)
        print('k=', k, '平均覆盖大小：', average_cover_size)

        list_IC_hep.append({
            'k': k,
            'run time': cal_time,
            'average cover size': average_cover_size,
            'S': S
        })
        temp_time = timer()  # 记录当前时间

    import pandas as pd

    df_IC_random_hep = pd.DataFrame(list_IC_hep)
    df_IC_random_hep.to_csv('../../data/output/singleDiscount/IC_singleDiscount_phy_Graph.csv')
    print('文件输出完毕——结束')
