"""
基于独立级联模型的degree heuristic[1]的实现
获取拥有最大度的前k个节点

[1] -- Wei Chen et al. Efficient influence maximization in Social Networks
"""
from algorithm.priorityQueue import PriorityQueue as PQ  # 优先队列

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
        degree = sum([G[u][v]['weight'] for v in G[u]])
        # degree = len(G[u])
        d.add_task(u, -degree)
    for i in range(k):
        u, priority = d.pop_item()
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
        degree = sum([G[u][v]['weight'] for v in G[u]])
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
    from algorithm.graph_data_handle import read_gpickle

    G = read_gpickle("../../data/graphs/hep.gpickle")
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    # 生成固定的传播概率
    # from algorithm.generation_propagation_probability import fixed_probability
    # Ep = fixed_probability(G, 0.01)

    I = 1000
    S = degreeHeuristic(G, 10)
    cal_time = time.time()
    print('算法运行时间：', cal_time - read_time)
    print('选取节点集为：', S)

    # from algorithm.IC.IC import avgIC_cover_size
    #
    # print('平均覆盖大小：', avgIC_cover_size(G, S, 0.01, I))
