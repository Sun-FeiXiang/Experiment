"""
基于独立级联模型random heuristic[1]
随机均匀取k个节点

[1] -- Wei Chen et al. Efficient influence maximization in Social Networks
"""
import networkx as nx
from runIAC import avgIAC


def randomHeuristic(G, k, p=.01):
    """
    在独立级联模型下找到初始传播的k个点
    输入: G -- networkx图对象
    k -- 需要的节点数
    p -- 传播概率
    输出:
    S -- 选择的k个点的集合
    """
    import random
    S = random.sample(G.nodes(), k)
    return S


if __name__ == "__main__":
    import time

    start = time.time()

    G_gpickle = nx.read_gpickle("../../data/graphs/hep.gpickle")
    print('Read graph G')
    read_time = time.time()
    print(read_time - start)

    # 获得传播概率
    Ep = dict()
    p = 0.01
    G = nx.DiGraph()
    for key, values in G_gpickle.edge.items():
        # print(key,list(values.keys()))
        for end in list(values.keys()):
            G.add_edge(key, end, weight=values[end]['weight'])
            Ep[(key, end)] = p

    I = 1000

    print('Calculate...')
    S = randomHeuristic(G, 10)
    cal_time = time.time()
    print(cal_time - read_time)
    print('节点集为：', S)
    print('平均覆盖大小：', avgIAC(G, S, Ep, I))
