"""
使用networkx网络结构的数据，进行传播
weight是传播概率
"""


def runIC(G, S, p=.01):
    """
    运行独立级联
    :param G: networkx图对象
    :param S: 种子集
    :param p: 传播概率
    :return:
    """
    from copy import deepcopy
    from random import random
    T = deepcopy(S)
    i = 0
    while i < len(T):
        for v in G[T[i]]:
            if v not in T:
                w = G[T[i]][v]['weight']  # 两个节点间边的数目
                if random() <= 1 - (1 - p) ** w:  # 如果至少一条边被影响
                    # print(T[i], 'influences', v)
                    T.append(v)
        i += 1
    return T


def runIC2(G, S, p=.01):
    """
    层次传播
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
        # print(i, Anext)
        i += 1
        T.extend(Acur)
        Anext = []
    return T


def spread_run_IC(G, S, p, iterations):
    avg = 0
    for i in range(iterations):
        avg += float(len(runIC(G, S, p))) / iterations
    return avg
