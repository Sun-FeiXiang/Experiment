"""
算法：基于独立级联模型的degree heuristic
    获取拥有最大度的前k个节点
来源：Wei Chen et al. Efficient influence maximization in Social Networks
"""
import collections
import copy

import networkx as nx

from algorithm.priorityQueue import PriorityQueue as PQ
import numpy as np
from preprocessing.generation_propagation_probability import p_fixed,p_random,p_fixed_with_link,p_inEdge,fixed_weight
from preprocessing.read_txt_nx import read_Graph
from model.ICM_nx import spread_run_IC,IC
import pandas as pd

def nodeDegree(G):
    D = {}
    for v in G:
        D[v] = G.degree(v)
    return D

def Seeding(G):
    W = []
    S = []
    for v in G:
        W.append(v)

    D_PQ = PQ()
    D = nodeDegree(G)
    for v in G:
        D_PQ.add_task(v, -D[v])

    while(len(W) != 0):
        u, priority = D_PQ.pop_item()
        if u not in W:
            continue
        S.append(u)
        W.remove(u)
        for v in G[u]:
            if v in W:
                W.remove(v)
    print("S" + str(S))
    return S

def getMostNeighborLabel(G, v, LP):
    adjLabels = collections.defaultdict(int)
    flag = 0
    for adj in G[v]:
        for label in LP[adj]:
            if label == -1:
                continue
            adjLabels[label] += 1
            flag = 1
    if flag == 0:
        return -1
    maxAdjLabels = max(adjLabels.values())
    return [i[0] for i in adjLabels.items() if i[1] == maxAdjLabels]

def getCommunity(G, LP):
    Com_num = 0
    Com_union = {}
    for v in G:
        for label in LP[v]:
            if label == -1:
                continue
            if label in Com_union:
                Com_union[label].append(v)
            else:
                Com_num = Com_num+1
                Com_union[label] = [v]

    print(Com_union)
    return Com_union

def Label_propagation(G, S):
    LP = {}
    V = []
    #用节点号代替标签号 独一无二
    for v in G:
        LP[v] = [-1]
        V.append(v)

    for u in S:
        LP[u].append(u)

    flag = 1
    while(flag == 1):
        flag = 0
        V_plu = np.random.permutation(V)
        #print(V_plu)
        for v in V_plu:
            mostLabels = getMostNeighborLabel(G,v, LP)

            if mostLabels == -1:
                continue

            for label in mostLabels:
                if label not in LP[v]:
                    LP[v].append(label)
                    flag = 1

    C = getCommunity(G, LP)
    return C


def Finding(G, k, C):
    C = sorted(C.items(), key=lambda item: len(item[1]), reverse=True)
    print(C)
    S = []
    for key, value in C:
        S.append(key)
        if (len(S) >= k):
            break
    return S

if __name__ == "__main__":
    # G = read_Graph("../data/graphdata/arenas-pgp.edges.txt",directed=False)
    G = nx.read_edgelist("../data/graphdata/email.txt", nodetype=int,create_using=nx.Graph)  # 其他数据集使用此方式读取
    fixed_weight(G)
    S = Seeding(G)
    C = Label_propagation(G, S)
    output = Finding(G, 51, C)
    print("种子集合：",S)
    I = 1000
    p = 0.05
    p_fixed_with_link(G,p)
    print("p=",p,",R,I=",I,"data=email,Graph")
    list_IC_hep = []
    for k in range(1, 51):
        S = output[:k]
        cur_spread = IC(G, S, I)
        sum_nodes = float(len(G.nodes))
        print('k = ', k, '选取节点集为：', S)
        print('k=', k, '平均覆盖大小：', cur_spread)
        list_IC_hep.append({
            'k': k,
            'average cover size': cur_spread,
            'S': S
        })
    df_IC_hep = pd.DataFrame(list_IC_hep)
    df_IC_hep.to_csv('../data/output/IMLPA/IC_LPA_(p=0.05R,I=1000)_email_Graph.csv')
    print('文件输出完毕——结束')
