"""
算法：IC模型中的greedy heuristic
来源：[1] -- Wei Chen et al. Efficient Influence Maximization in Social Networks (Algorithm 2)

无向图

"""
from __future__ import division
from copy import deepcopy
import random
import networkx as nx
from timeit import default_timer as timer


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


def findCCs(G):
    # 从图G中移除阻塞边，获得传播图
    E = deepcopy(G)
    edge_rem = [e for e in E.edges() if random.random() < (1 - E[e[0]][e[1]]['weight'])]
    E.remove_edges_from(edge_rem)
    # 初始化 CC
    CCs = dict()  # 每个组件都反映了组件的成员数
    # BFS获得CCs
    for node in E.nodes():
        CCs[node] = bfs2(E, node)
    return CCs


def newGreedyIC(G, k, R=20):
    S = []
    for i in range(k):
        # print('k=',i)
        scores = {v: 0 for v in G}
        for j in range(R):
            CCs = findCCs(G)
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
    G = nx.read_weighted_edgelist("../../data/graphdata/phy.txt", comments='#', nodetype=int, create_using=nx.Graph())
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    # 生成固定的传播概率0.01
    from generation.generation_propagation_probability import weight_probability_fixed

    weight_probability_fixed(G)

    I = 1000
    list_IC_random_hep = []
    temp_time = timer()
    for k in range(1, 51):
        S = newGreedyIC(G, k)
        cal_time = timer() - temp_time
        print('newGreedyIC算法运行时间：', cal_time)
        print('k = ', k, '选取节点集为：', S)

        from algorithm.Spread.Networkx_spread import spread_run_IC

        average_cover_size = spread_run_IC(S, G, 1000)
        print('k=', k, '平均覆盖大小：', average_cover_size)

        list_IC_random_hep.append({
            'k': k,
            'run time': cal_time,
            'average cover size': average_cover_size,
            'S': S
        })
        temp_time = timer()  # 记录当前时间

    import pandas as pd

    df_IC_random_hep = pd.DataFrame(list_IC_random_hep)
    df_IC_random_hep.to_csv('../../data/output/greedy/IC_newGreedyIC_phy_Graph.csv')
    print('文件输出完毕——结束')
