"""
影响传播的独立级联模型
"""


def runIC(G, S, p=.01):
    """
    输入: G -- networkx图对象
    S -- 顶点的初始集合
    p -- 传播概率
    输出: T -- 受到影响的节点集（包括S）
    """
    from copy import deepcopy
    from random import random
    T = deepcopy(S)  # 复制初始节点集

    # ugly C++ version
    i = 0
    while i < len(T):
        for v in G[T[i]]:  # 已传播点的邻居节点
            if v not in T:  # 如果它还没有被激活
                w = G[T[i]][v]['weight']  # 计算两个点之间的边数
                if random() <= 1 - (1 - p) ** w:  # 如果至少一条边传播影响
                    # print(T[i], 'influences', v)
                    T.append(v)
        i += 1
    # 整洁的 python ic 版本
    # legitimate version with dynamically changing list: http://stackoverflow.com/a/15725492/2069858
    # for u in T: # T may increase size during iterations
    #     for v in G[u]: # check whether new node v is influenced by chosen node u
    #         w = G[u][v]['weight']
    #         if v not in T and random() < 1 - (1-p)**w:
    #             T.append(v)
    return T


def runIC2(G, S, p=.01):
    """ Runs independent cascade model (finds levels of propagation).
    Let A0 be S. A_i is defined as activated nodes at ith step by nodes in A_(i-1).
    We call A_0, A_1, ..., A_i, ..., A_l levels of propagation.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    """
    from copy import deepcopy
    import random
    T = deepcopy(S)
    Acur = deepcopy(S)
    Anext = []
    i = 0
    while Acur:
        values = dict()
        for u in Acur:
            for v in G[u]:
                if v not in T:
                    w = G[u][v]['weight']
                    if random.random() < 1 - (1 - p) ** w:
                        Anext.append((v, u))
        Acur = [edge[0] for edge in Anext]
        i += 1
        T.extend(Acur)
        Anext = []
    return T


def avgIC_cover_size(G, S, p, iterations):
    """
    :param G: 图
    :param S: 初始集合
    :param p: 传播概率
    :param iterations: 迭代次数
    :return: 平均覆盖大小
    """
    avg = 0
    for i in range(iterations):
        avg += float(len(runIC(G, S, p))) / iterations
    return avg
