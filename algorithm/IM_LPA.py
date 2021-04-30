"""
算法：基于独立级联模型的degree heuristic
    获取拥有最大度的前k个节点
来源：Wei Chen et al. Efficient influence maximization in Social Networks
"""
import collections
import copy


from algorithm.priorityQueue import PriorityQueue as PQ
import numpy as np

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
    print(S)
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

    from dataPreprocessing.read_txt_nx import read_Graph

    G = read_Graph("../data/graphdata/celegans.txt")
    S = Seeding(G)
    C = Label_propagation(G, S)
    output = Finding(G, 90, C)
    print(S)
    from diffusion.Networkx_diffusion import spread_run_IC
    #average_cover_size = spread_run_IC(G, S, 0.05, 1000)
    #print(average_cover_size)


    list_IC_hep = []
    for k in range(1, 30):
        S = output[:k]
        cur_spread = spread_run_IC(G, S, 0.25, 1000)
        print('k = ', k, '选取节点集为：', S)
        print('k=', k, '平均覆盖大小：', cur_spread)
        list_IC_hep.append({
            'k': k,
            'average cover size': cur_spread,
            'S': S
        })
    import pandas as pd

    df_IC_hep = pd.DataFrame(list_IC_hep)
    df_IC_hep.to_csv('../data/output/IMLPA/IC_node=16_(p=0.25)_celegans_10000_Graph.csv')
    # print('文件输出完毕——结束')
