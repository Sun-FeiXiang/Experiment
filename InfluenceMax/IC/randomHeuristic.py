"""
基于独立级联模型random heuristic[1]
随机均匀取k个节点

[1] -- Wei Chen et al. Efficient influence maximization in Social Networks
"""


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
