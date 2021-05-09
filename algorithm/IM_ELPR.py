"""

"""
import collections
import copy
from algorithm.priorityQueue import PriorityQueue as PQ
import numpy as np
from preprocessing.read_txt_nx import read_Graph
from model.ICM_nx import spread_run_IC,IC

def eh_index(G):
    EH = {}
    H = {}
    for v in G:
        h_index = G.degree(v)
        for h in range(0, h_index + 1):
            ch = 0
            for u in G[v]:
                if h <= G.degree(u):
                    ch = ch + 1
            if ch < h:
                h_index = h - 1
                break
        H[v] = h_index
    for v in G:
        EH[v] = H[v]
        for u in G[v]:
            EH[v] = EH[v] + H[u]
    return EH


def Seeding(G):
    W = []
    S = []
    for v in G:
        W.append(v)

    EH_PQ = PQ()
    EH = eh_index(G)
    for v in G:
        EH_PQ.add_task(v, -EH[v])

    while (len(W) != 0):
        u, priority = EH_PQ.pop_item()
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
                Com_num = Com_num + 1
                Com_union[label] = [v]

    print(Com_union)
    return Com_union


def Label_propagation(G, S):
    LP = {}
    V = []
    # 用节点号代替标签号 独一无二
    for v in G:
        LP[v] = [-1]
        V.append(v)

    for u in S:
        LP[u].append(u)

    flag = 1
    while flag == 1:
        flag = 0
        V_plu = np.random.permutation(V)
        # print(V_plu)
        for v in V_plu:
            mostLabels = getMostNeighborLabel(G, v, LP)

            for label in mostLabels:
                if label not in LP[v]:
                    LP[v].append(label)
                    flag = 1

    C = getCommunity(G, LP)
    return C


def Com_edge(G, C, i, j):
    Sum_edge = 0
    for v in C[i]:
        for u in G[v]:
            if u in C[j] and u not in C[i]:
                Sum_edge = Sum_edge + 1
    return Sum_edge


def modularity(g, community_list):
    # ls, ds variables
    intra_degree = {i: 0 for i in range(0, len(community_list))}  # ds
    intra_edges = {i: 0 for i in range(0, len(community_list))}  # ls

    # calculate ds, time complexity: O(V)
    community_index = 0
    community_id = {}

    for key, community in community_list.items():
        tmp_index = copy.copy(community_index)
        for v in community:
            intra_degree[tmp_index] += g.degree(v)
            community_id[v] = tmp_index
        community_index += 1

    # calculate ls, time complexity: O(E)

    for (ei, ej) in g.edges():
        if community_id[ei] == community_id[ej]:
            intra_edges[community_id[ei]] += 1
        else:
            pass

    # calculate modularity Q, time complexity: O(C)
    q = 0
    num_edges = g.number_of_edges()

    for i in range(0, len(community_list)):
        ls = intra_edges[i] / num_edges
        ds = pow((intra_degree[i] / (2 * num_edges)), 2)
        q += (ls - ds)

    return q


def Merge_Community(G, C):
    n = len(G)
    R = np.zeros((n + 1, n + 1))
    flag = 1
    while (flag):
        for i in C:
            for j in C:
                if i == j:
                    R[i][j] = Com_edge(G, C, i, j)
                else:
                    x = Com_edge(G, C, i, j)
                    y = min(len(C[i]), len(C[j]))
                    R[i][j] = x * 1.0 / y
        new_C = copy.deepcopy(C)
        maxi = 0
        maxj = 0
        maxR = 0
        for i in C:
            for j in C:
                if i == j:
                    continue
                if R[i][j] > maxR:
                    max = R[i][j]
                    maxi = i
                    maxj = j
        for v in new_C[maxj]:
            new_C[maxi].append(v)
        new_C.pop(maxj)

        new_modularity = modularity(G, new_C)
        pre_modularity = modularity(G, C)

        if new_modularity > pre_modularity:
            C = copy.deepcopy(new_C)
        else:
            flag = 0

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

    G = read_Graph("../data/graphdata/hep.txt")
    S = Seeding(G)
    C = Label_propagation(G, S)
    C = Merge_Community(G, C)
    output = Finding(G, 50, C)
    print(S)

    # average_cover_size = spread_run_IC(G, S, 0.05, 1000)
    # print(average_cover_size)

    list_IC_hep = []
    for k in range(1, 51):
        S = output[:k]
        cur_spread = spread_run_IC(G, S, 0.01, 1000)
        print('k = ', k, '选取节点集为：', S)
        print('k=', k, '平均覆盖大小：', cur_spread)
        list_IC_hep.append({
            'k': k,
            'average cover size': cur_spread,
            'S': S
        })
    # import pandas as pd
    #
    # df_IC_hep = pd.DataFrame(list_IC_hep)
    # df_IC_hep.to_csv('../data/output/IMELPR/IC_node=90(p=0.25)_celegans_10000_Graph.csv')
    # print('文件输出完毕——结束')
