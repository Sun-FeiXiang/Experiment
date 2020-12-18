"""
算法：IC模型中的greedy heuristic
来源：[1] -- Wei Chen et al. Efficient Influence Maximization in Social Networks (Algorithm 2)

无向图

"""
from __future__ import division
from copy import deepcopy
import random

def bfs(E, S):
    """
    使用BFS找到图E中子集S的所有可能到达的顶点集
    输入: E -- networkx图对象
    S -- 初始节点集
    输出: Rs -- S可以到达的节点集
    """
    Rs = []
    for u in S:
        if u in E:
            if u not in Rs:
                Rs.append(u)
            for v in E[u].keys():
                if v not in Rs:
                    Rs.append(v)
    return Rs


def bfs2(E, node):
    """
    :param E: 传播图
    :param node: 节点node
    :return: node在E中可到达的节点集
    """
    visited = set()
    import queue
    q = queue.Queue()
    q.put(node)
    res = []
    while not q.empty():
        u = q.get()
        res.append(u)
        adj = list(E.adj[u].keys())
        if len(adj) != 0:
            for v in adj:
                if v not in visited:
                    visited.add(v)
                    q.put(v)
    return res


def findCCs(G, Ep):
    # 从图G中移除阻塞边，获得传播图
    E = deepcopy(G)
    edge_rem = [e for e in E.edges() if random.random() < (1 - Ep[e]) ** (E[e[0]][e[1]]['weight'])]
    E.remove_edges_from(edge_rem)
    # 初始化 CC
    CCs = dict()  # 每个组件都反映了组件的成员数
    # BFS获得CCs
    for node in E.nodes():
        CCs[node] = bfs2(E, node)
    return CCs


def newGreedyIC(G, k, Ep, R=20):
    S = []
    for i in range(k):
        # print('k=',i)
        scores = {v: 0 for v in G}
        for j in range(R):
            CCs = findCCs(G, Ep)
            for v in CCs:
                if v not in S:
                    scores[v] += float(len(CCs[v])) / R
        max_v = sorted(scores,key=lambda x:scores[x])[-1]
        max_score = scores[max_v]
        S.append(max_v)
    return S


if __name__ == "__main__":
    import time

    start = time.time()
    from data_handle.graph_data_handle import read_gpickle_DiGraph

    G = read_gpickle_DiGraph("../../data/graphs/hep.gpickle")
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    # 生成固定的传播概率
    from generation.generation_propagation_probability import fixed_probability

    Ep = fixed_probability(G, 0.01)

    I = 1000
    S = newGreedyIC(G, 5, Ep)
    cal_time = time.time()
    print('算法运行时间：', cal_time - read_time)
    print('选取节点集为：', S)

    from algorithm.IC.IC import avgIC_cover_size

    print('平均覆盖大小：', avgIC_cover_size(G, S, 0.01, I))
