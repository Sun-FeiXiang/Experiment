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


'''
File for Linear Threshold model (LT).
For directed graph G = (nodes, edges, weights) and
set of thresholds lambda for each node, LT model works as follows:
Initially set S of nodes is activated. For all outgoing neighbors,
we compare sum of edge weights from activated nodes and node thresholds.
If threshold becomes less than sum of edge weights for any given vertex,
then this vertex becomes active for the following iterations.
LT stops when no activation happens.

More on this: Kempe et al."Maximizing the spread of influence in a social network"
'''
import random
from copy import deepcopy
import networkx as nx

def uniformWeights(G):
    '''

    '''
    Ew = dict()
    for u in G:
        in_edges = G.in_edges([u], data=True)
        dv = sum([edata['weight'] for v1,v2,edata in in_edges])
        for v1,v2,_ in in_edges:
            Ew[(v1,v2)] = 1/dv
    return Ew

def randomWeights(G):
    '''
    Every edge has random weight.
    After weights assigned,
    we normalize weights of all incoming edges so that they sum to 1.
    '''
    Ew = dict()
    for u in G:
        in_edges = G.in_edges([u], data=True)
        ew = [random.random() for e in in_edges] # random edge weights
        total = 0 # total sum of weights of incoming edges (for normalization)
        for num, (v1, v2, edata) in enumerate(in_edges):
            total += edata['weight']*ew[num]
        for num, (v1, v2, _) in enumerate(in_edges):
            Ew[(v1,v2)] = ew[num]/total
    return Ew

def checkLT(G, Ew, eps = 1e-4):
    ''' To verify that sum of all incoming weights <= 1
    '''
    for u in G:
        in_edges = G.in_edges([u], data=True)
        total = 0
        for (v1, v2, edata) in in_edges:
            total += Ew[(v1, v2)]*G[v1][v2]['weight']
        if total >= 1 + eps:
            return 'For node %s LT property is incorrect. Sum equals to %s' %(u, total)
    return True

def runLT(G, S, p):
    '''
    Input: G -- networkx directed graph
    S -- initial seed set of nodes
    Ew -- influence weights of edges
    NOTE: multiple k edges between nodes (u,v) are
    considered as one node with weight k. For this reason
    when u is activated the total weight of (u,v) = Ew[(u,v)]*k
    '''

    # assert type(G) == nx.DiGraph, 'Graph G should be an instance of networkx.DiGraph'
    # assignedert type(S) == list, 'Seed set S should be an instance of list'
    #assert type(Ew) == dict, 'Infleunce edge weights Ew should be an instance of dict'

    T = deepcopy(S) # 种子集
    lv = dict() # 每个节点的阈值
    for u in G:
        lv[u] = random.random()

    W = dict(zip(G.nodes(), [0]*len(G))) # weighted number of activated in-neighbors
    Sj = deepcopy(S)
    # print 'Initial set', Sj
    while len(Sj): # while we have newly activated nodes
        Snew = []
        for u in Sj:
            for v in G[u]:
                if v not in T:
                    W[v] += p*G[u][v]['weight']
                    if W[v] >= lv[v]:
                        # print 'Node %s is targeted' %v
                        Snew.append(v)
                        T.append(v)
        Sj = deepcopy(Snew)
    return T

def spread_run_LT(G, S, p, iterations):
    avgSize = 0
    progress = 1
    for i in range(iterations):
        # if i == round(iterations*.1*progress) - 1:
        #     print (10*progress, '% done')
        #     progress += 1
        T = runLT(G, S, p)
        avgSize += len(T)/iterations

    return avgSize


"""
触发模型

"""