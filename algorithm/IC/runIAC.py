"""
独立任意级联模型
Independent Arbitrary Cascade (IAC) 是拥有任意传播概率的独立级联模型。
"""

from __future__ import division

import math
import random
from copy import deepcopy

import networkx as nx


def uniformEp(G, p=.01):
    """
    每条边都拥有相同的概率p.
    """
    if type(G) == type(nx.DiGraph()):
        Ep = dict(zip(G.edges(), [p] * len(G.edges())))
    elif type(G) == type(nx.Graph()):
        Ep = dict()
        for (u, v) in G.edges():
            Ep[(u, v)] = p
            Ep[(u, v)] = p
    else:
        raise ValueError("Provide either nx.Graph or nx.DiGraph object")
    return Ep


def randomEp(G, maxp):
    """
    Every edge has random propagation probability <= maxp <= 1
    """
    assert maxp <= 1, "Maximum probability cannot exceed 1."
    Ep = dict()
    if type(G) == type(nx.DiGraph()):
        for v1, v2 in G.edges():
            p = random.uniform(0, maxp)
            Ep[(v1, v2)] = p
    elif type(G) == type(nx.Graph()):
        for v1, v2 in G.edges():
            p = random.uniform(0, maxp)
            Ep[(v1, v2)] = p
            Ep[(v2, v1)] = p
    else:
        raise ValueError("Provide either nx.Graph or nx.DiGraph object")
    return Ep


def random_from_range(G, prange):
    '''
    Every edge has propagation probability chosen from prange uniformly at random.
    '''
    for p in prange:
        if p > 1:
            raise ValueError("Propagation probability inside range should be <= 1")
    Ep = dict()
    if type(G) == type(nx.DiGraph()):
        for v1, v2 in G.edges():
            p = random.choice(prange)
            Ep[(v1, v2)] = p
    elif type(G) == type(nx.DiGraph()):
        for v1, v2 in G.edges():
            p = random.choice(prange)
            Ep[(v1, v2)] = p
            Ep[(v2, v1)] = p
    return Ep


# http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
def chunks(lst, n):
    size = int(math.ceil(float(len(lst)) / n))
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def degree_categories(G, prange):
    """
    每一条边都有一个基于节点度数的传播概率。
    """
    for p in prange:
        if p > 1:
            raise ValueError("Propagation probability inside range should be <= 1")
    Ep = dict()

    d = {v: sum([G[v][u]["weight"] for u in G[v]]) for v in G}
    sorted_d = chunks(sorted(d.items(), key=lambda _, degree: degree), len(prange))
    sorted_p = sorted(prange)
    categories = zip(sorted_p, sorted_d)
    dp = dict()
    for c in categories:
        p, nodes = c
        for (v, _) in nodes:
            dp[v] = p

    if type(G) == type(nx.DiGraph()):
        for v1, v2 in G.edges():
            Ep[(v1, v2)] = dp[v1]
    elif type(G) == type(nx.DiGraph()):
        for v1, v2 in G.edges():
            Ep[(v1, v2)] = dp[v2]
            Ep[(v2, v1)] = dp[v1]
    return Ep


def weightedEp(G):
    """
    v的每一个输入边的传播概率等于1/deg（v）
    """
    Ep = dict()
    for v in G:
        in_edges = G.in_edges([v])
        degree = sum([G[u][v]["weight"] for (u, _) in in_edges])
        for edge in in_edges:
            Ep[edge] = 1.0 / degree
    return Ep


def runIAC(G, S, Ep):
    """
    运行独立级联模型
    Input: G -- networkx图
    S -- 输出话节点集
    Ep -- 传播概率
    Output: T -- 影响的节点集（包括S）

    NOTE:
    Ep是一个字典，它对应于每个边的关联概率如果每个边（v1，v2）的图是无向的，概率为p，
    我们有Ep[(v1，v2)]=p，Ep[(v2，v1)]=p。
    """
    T = deepcopy(S)  # 复制已经存在的节点

    # ugly C++ version
    i = 0
    while i < len(T):
        for v in G[T[i]]:  # for neighbors of a selected node
            if v not in T:  # if it wasn't selected yet
                w = G[T[i]][v]['weight']  # count the number of edges between two nodes
                p = Ep[(T[i], v)]  # propagation probability
                if random.random() <= 1 - (1 - p) ** w:  # if at least one of edges propagate influence
                    # print T[i], 'influences', v
                    T.append(v)
        i += 1
    return T


def avgIAC(G, S, Ep, I):
    """
    Input:
        G -- 无向图
        S -- 节点集
        Ep -- 传播概率
        I -- 迭代次数
    Output:
        avg -- 平均覆盖大小
    """
    avg = 0
    for i in range(I):
        avg += float(len(runIAC(G, S, Ep))) / I
    return avg


def findCC(G, Ep):
    # remove blocked edges from graph G
    E = deepcopy(G)
    edge_rem = [e for e in E.edges() if random.random() < (1 - Ep[e]) ** (E[e[0]][e[1]]['weight'])]
    E.remove_edges_from(edge_rem)

    # initialize CC
    CC = dict()  # number of a component to its members
    explored = dict(zip(E.nodes(), [False] * len(E)))
    c = 0
    # perform BFS to discover CC
    for node in E:
        if not explored[node]:
            c += 1
            explored[node] = True
            CC[c] = [node]
            component = E[node].keys()
            for neighbor in component:
                if not explored[neighbor]:
                    explored[neighbor] = True
                    CC[c].append(neighbor)
                    component.extend(E[neighbor].keys())
    return CC


def findL(CCs, T):
    # find top components that can reach T activated nodes
    sortedCCs = sorted([(len(dv), dk) for (dk, dv) in CCs.iteritems()], reverse=True)
    cumsum = 0  # sum of top components
    L = 0  # current number of CC that achieve T
    # find L first
    for length, numberCC in sortedCCs:
        L += 1
        cumsum += length
        if cumsum >= T:
            break
    return L, sortedCCs


def findCCs_size_distribution(G, Ep, T):
    CCs = findCC(G, Ep)
    L, sortedCCs = findL(CCs, T)
    from itertools import groupby
    histogram = [(s, len(list(group))) for (s, group) in groupby(sortedCCs, key=lambda size, _: size)]

    bluedots = 1
    acc_size = 0
    for (size, number) in histogram:
        acc_size += size
        if acc_size < T:
            bluedots += 1
        else:
            break

    return histogram, bluedots, L, len(CCs)


def findLrangeforTrange(G, Ep, Trange):
    Lrange = []
    CCs = findCC(G, Ep)
    for T in Trange:
        L, _ = findL(CCs, T)
        Lrange.append(L)
    return Lrange, len(CCs)


if __name__ == '__main__':
    import time

    start = time.time()
