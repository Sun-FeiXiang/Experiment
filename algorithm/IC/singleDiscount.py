"""
算法：基于独立级联的简单折扣启发式算法 single discount heuristic[1]

参考：[1] -- Wei Chen et al. Efficient influence maximization in Social Networks
"""
from priorityQueue import PriorityQueue as PQ  # priority queue


def singleDiscount(G, k, p=.1):
    """
    在独立级联模型中查找要传播的初始节点集（带优先级队列）
    Input: G -- networkx图对象
    k -- 需要的节点数
    p -- 传播概率
    Output:
    S -- 选择的k个点的集合
    """
    S = []  # 激活的节点集
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
