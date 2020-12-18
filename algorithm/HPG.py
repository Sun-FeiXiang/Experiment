"""
来源：（HPG）一种新型的社会网络影响最大化算法
2011年 计算机学报 中文核心 A类
"""
from algorithm.priorityQueue import PriorityQueue as PQ  # priority queue
import math
from algorithm.IC.IC import runIC


def buv(G, u, v):
    """
    无权图是b_uv的计算方式。
    :param G:
    :param u:
    :param v:
    :return:
    """
    outDeg_u = len(G.adj[u])
    outDeg_v = 0
    for w in G.adj[v]:
        outDeg_v = outDeg_v + len(G.adj[w])
    return outDeg_u / outDeg_v


def PI(G, u, S):
    inf = dict()
    inf[u] = 0
    t = 0
    for v in G[u]:
        t = t + 1
        if v not in S:
            inf[u] += buv(G, u, v)

    return t + (1 - math.exp(-inf[u]))


def HPG(G, k, c=0.5):
    """
    在独立级联模型中查找要传播的初始节点集（带优先级队列）
    Input:
    G -- networkx 图对象
    k -- 需要的节点个数
    c -- 启发因子
    Output:
    S -- 选择的k个点的集合
    """

    k1 = k - math.ceil(c * k)
    k2 = math.ceil(c * k)

    S = []
    T = []  # 被激活的节点集

    dd = PQ()
    inf = dict()
    pi = dict()

    for u in G.nodes():
        pi[u] = PI(G, u, S)
        dd.add_task(u, pi[u])  # 添加每个节点的度数

    for i in range(k1):
        u, priority = dd.pop_item()  # 从未激活的节点中选择PI最大的节点
        S.append(u)
        T.append(u)
        dd = PQ()
        T = runIC(G, T)
        for u in G.nodes():
            if u not in T:
                pi[u] = PI(G, u, T)
                dd.add_task(u, pi[u])

    R = 20
    for i in range(k2):
        s = PQ()  # 优先队列
        for v in G.nodes():
            if v not in T:
                s.add_task(v, 0)  # 初始传播值
                for j in range(R):  # 运行R次随机级联
                    [priority, count, task] = s.entry_finder[v]
                    s.add_task(v, priority - float(len(runIC(G, T + [v]))) / R)  # 加入标准传播值
        task, priority = s.pop_item()
        S.append(task)
        T.append(task)
        T = runIC(G, T)
    return S


if __name__ == "__main__":
    import time

    start = time.time()
    from data_handle.graph_data_handle import read_gpickle_DiGraph

    G = read_gpickle_DiGraph("../data/graphs/hep.gpickle")
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    # 生成固定的传播概率
    # from algorithm.generation_propagation_probability import fixed_probability
    # Ep = fixed_probability(G, 0.01)

    I = 1000
    S = HPG(G, 10)
    cal_time = time.time()
    print('算法运行时间：', cal_time - read_time)
    print('选取节点集为：', S)

    from algorithm.IC.IC import avgIC_cover_size

    print('平均覆盖大小：', avgIC_cover_size(G, S, 0.01, I))
