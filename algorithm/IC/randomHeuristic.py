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
    from algorithm.graph_data_handle import read_gpickle

    G = read_gpickle("../../data/graphs/hep.gpickle")
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    # 生成固定的传播概率
    # from algorithm.generation_propagation_probability import fixed_probability
    # Ep = fixed_probability(G, 0.01)

    I = 1000
    S = randomHeuristic(G, 10)
    cal_time = time.time()
    print('算法运行时间：', cal_time - read_time)
    print('选取节点集为：', S)

    from algorithm.IC.IC import avgIC_cover_size

    print('平均覆盖大小：', avgIC_cover_size(G, S, 0.01, I))
