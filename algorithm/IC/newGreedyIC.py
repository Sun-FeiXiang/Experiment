"""
Implements greedy heuristic for IC model [1]
[1] -- Wei Chen et al. Efficient Influence Maximization in Social Networks (Algorithm 2)
"""
from __future__ import division
from copy import deepcopy  # copy graph object
import random
import networkx as nx
from runIAC import avgIAC
import matplotlib.pyplot as plt

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

def bfs2(E,node):
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
        CCs[node] = bfs2(E,node)
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
        max_v, max_score = max(scores.items())
        S.append(max_v)
    return S


# def newGreedyIC(G, k, p=.01):
#     ''' Finds initial set of nodes to propagate in Independent Cascade.
#     Input: G -- networkx graph object
#     k -- number of nodes needed
#     p -- propagation probability
#     Output: S -- set of k nodes chosen
#     '''
#
#     import time
#     start = time.time()
#
#     # assert type(S0) == list, "S0 must be a list. %s provided instead" % type(S0)
#     # S = S0 # set of selected nodes
#     # if len(S) >= k:
#     #     return S[:k]
#
#     S = []
#
#     iterations = k - len(S)
#     print 'iterations =', iterations
#     for i in range(iterations):
#         # s = PQ() # number of additional nodes each remained mode will bring to the set S in R iterations
#         s = dict()
#         Rv = dict() # number of reachable nodes for node v
#         # initialize values of s
#         for v in G:
#             if v not in S:
#                 # s.add_task(v, 0)
#                 s[v] = 0
#
#         # calculate potential additional spread for each vertex not in S
#         prg_idx = 1
#         idx = 1
#         prcnt = .1 # for progress to print
#         R = 20 # number of iterations to run RanCas
#         # spread from each node individually in pruned graph E
#         # Rv = dict()
#         # for v in G:
#         #     if v not in S:
#         #         Rv[v] = 0
#         for j in range(R):
#             # create new pruned graph E
#             E = deepcopy(G)
#             edge_rem = [] # edges to remove
#             for (u,v) in E.edges():
#                 w = G[u][v]['weight']
#                 if random() < (1 - p)**w:
#                     edge_rem.append((u,v))
#             E.remove_edges_from(edge_rem)
#             # find reachable vertices from S
#             # TODO make BFS happens only once for all nodes. Should take O(m) time.
#             Rs = bfs(E, S)
#             # find additional nodes each vertex would bring to the set S
#             time2update = time.time()
#             for v in G:
#                 if v not in S + Rs: # if node has not chosen in S and has chosen by spread from S
#                     # Rv[v] += float(len(bfs(E, [v])))/R
#                     # [priority, c, task] = s.entry_finder[v]
#                     # s.add_task(v, priority - float(len(bfs(E, [v])))/R)
#                     s[v] -= float(len(bfs(E, [v])))/R
#             # print 'Took %s sec to update' %(time.time() - time2update)
#
#             if idx == int(prg_idx*prcnt*R):
#                 print '%s%%...' %(int(prg_idx*prcnt*100)), time.time() - start
#                 prg_idx += 1
#             idx += 1
#         # add spread of nodes in G'
#         # for v in Rv:
#         #     s.add_task(v, -Rv[v])
#         # add vertex with maximum potential spread
#         time2min = time.time()
#         # task, priority = s.pop_item()
#         task, priority = min(s.iteritems(), key=lambda (dk,dv): dv)
#         s.pop(task)
#         # print 'Took %s sec to find min' %(time.time() - time2min)
#         S.append(task)
#         print i, k, task, -priority, time.time() - start
#     return S

if __name__ == "__main__":
    import time

    start = time.time()

    G_gpickle = nx.read_gpickle("../../data/graphs/hep.gpickle")
    print('Read graph G')
    print(time.time() - start)

    model = "MultiValency"
    ep_model = ""
    if model == "MultiValency":
        ep_model = "range"
    elif model == "Random":
        ep_model = "random"
    elif model == "Categories":
        ep_model = "degree"

    # 获得传播概率
    Ep = dict()
    p = 0.01
    G = nx.DiGraph()
    for key,values in G_gpickle.edge.items():
        # print(key,list(values.keys()))
        for end in list(values.keys()):
            G.add_edge(key,end,weight=values[end]['weight'])
            Ep[(key,end)] = p

    I = 1000

    S = newGreedyIC(G, 10, Ep)
    print('节点集为：',S)
    print('平均覆盖大小：',avgIAC(G, S, Ep, I))
