"""
算法：IC模型中的newGreedyIC
来源：[1] -- Wei Chen et al. Efficient Influence Maximization in Social Networks (Algorithm 3)

无向图

"""
from __future__ import division
from copy import deepcopy
import random
from timeit import default_timer as timer
from model.ICM_nx import IC
from time import time
from preprocessing.read_txt_nx import read_Graph
from preprocessing.generation_propagation_probability import p_fixed,fixed_weight

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
    edge_rem = [e for e in E.edges() if random.random() < (1 - G.edges[e]['p']) ** (E[e[0]][e[1]]['weight'])]
    E.remove_edges_from(edge_rem)
    # 初始化 CC
    CCs = dict()  # each component is reflection of the number of a component to its members
    explored = dict(zip(E.nodes(), [False] * len(E)))
    c = 0
    # perform BFS to discover CC
    for node in E:
        if not explored[node]:
            c += 1
            explored[node] = True
            CCs[c] = [node]
            component = list(E[node].keys())
            for neighbor in component:
                if not explored[neighbor]:
                    explored[neighbor] = True
                    CCs[c].append(neighbor)
                    component.extend(E[neighbor].keys())
    return CCs


def newGreedyIC(G, k, R=10000):
    S, spread, timelapse, start_time = [], [], [], timer()
    for i in range(k):
        scores = {v: 0 for v in G}
        for j in range(R):
            CCs = findCCs(G)
            # print(CCs)
            for CC in CCs.values():
                # print(CC)
                for v in S:
                    if v in CC:
                        break
                else:  # in case CC doesn't have node from S
                    for u in CC:
                        scores[u] += float(len(CC)) / R
        max_v, max_score = max(scores.items(), key=lambda x: x[1])
        # print(max_v, max_score)
        S.append(max_v)
        cal_time = timer() - start_time
        timelapse.append(cal_time)
    return (S, spread, timelapse)


if __name__ == "__main__":
    start = time()
    G = read_Graph('../../data/graphdata/hep.txt',directed=False)
    read_time = time()
    print('读取网络时间：', read_time - start)
    p = 0.01
    I = 1000
    # p_fixed_with_link(G, p)
    p_fixed(G, p)
    # p_random(G)
    # p_inEdge(G)
    list_IC_hep = []
    out_put = newGreedyIC(G, 50)
    print("p=", p, ",I=", I, ",data=hep,Graph")
    for k in range(1, 51):
        S = out_put[0][:k]
        cur_spread = IC(G, S, I)
        cal_time = out_put[1][k - 1]
        print('newGreedyIC算法运行时间：', cal_time)
        print('k = ', k, '选取节点集为：', S)
        print('k=', k, '平均覆盖大小：', cur_spread)
        list_IC_hep.append({
            'k': k,
            'run time': cal_time,
            'average cover size': cur_spread,
            'S': S
        })
    import pandas as pd

    df_IC_random_hep = pd.DataFrame(list_IC_hep)
    df_IC_random_hep.to_csv('../../data/output/greedy/IC_newGreedyIC(p=0.01)_hep_Graph.csv')
    print('文件输出完毕——结束')
