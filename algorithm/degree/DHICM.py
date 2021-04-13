"""
来源：Efficient Influence Maximization in Social-Networks Under Independent Cascade Model
2020
本文无重大改进，但对独立级联模型的传播概率进行了“加强”
???实验结果与论文不同~

"""
import time
import networkx as nx


def DHICM(G, k, Ep):
    d = dict()
    dd = dict()  # 度折扣
    t = dict()  # 选择邻居的个数
    S, timelapse, start_time = [], [], time.time()
    for u in G:
        d[u] = sum([G[u][v]['weight'] for v in G[u]])  # 每条边增加的度 一般是1，也有两个点之间有多条边
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd[u] = d[u]
        t[u] = 0
    for i in range(k):
        u, ddv = max(dd.items())
        dd.pop(u)
        S.append(u)
        timelapse.append(time.time() - start_time)
        for v in G[u]:
            if v not in S:
                dd[v] = d[v] - 1 - (d[u] - 1) * Ep[(u, v)]
    return (S, timelapse)


def runIC(G, S, Ep):
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
                if random() <= 1 - (1 - Ep[(T[i], v)]) ** w:  # 如果至少一条边被影响
                    # print(T[i], 'influences', v)
                    T.append(v)
        i += 1
    return T


def spread_run_IC(G, S, Ep, iterations):
    avg = 0
    for i in range(iterations):
        avg += float(len(runIC(G, S, Ep))) / iterations
    return avg


def fixed_probability(G, p, d, n):
    """
    :param G: 图
    :param p: 固定概率
    :return: 字典，每条边：概率
    """
    fp = dict()
    for edge in G.edges:
        fp[edge] = p + (d[edge[0]] + d[edge[1]]) / n + (len(set([G[edge[0]][v]['weight'] for v in G[edge[0]]]) & set([G[edge[1]][v]['weight'] for v in G[edge[1]]]))) / n
        fp[(edge[1],edge[0])] = fp[edge]#无向图
    return fp


if __name__ == "__main__":
    start = time.time()
    from dataPreprocessing.read_txt_nx import read_Graph

    G = read_Graph('../../data/graphdata/hep.txt')
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    # 强化传播概率
    p = 0.01
    d = dict()
    for u in G:
        d[u] = sum([G[u][v]['weight'] for v in G[u]])
    n = nx.number_of_nodes(G)  # 节点个数
    Ep = fixed_probability(G, p, d, n)

    # 算法运行
    output = DHICM(G, 50, Ep)

    list_IC_hep = []
    for k in range(1, 51):
        S = output[0][:k]
        cur_spread = spread_run_IC(G, S, Ep, 10000)
        cal_time = output[1][k - 1]
        print('DHICM算法运行时间：', cal_time)
        print('k = ', k, '选取节点集为：', S)
        print('k=', k, '平均覆盖大小：', cur_spread)
        list_IC_hep.append({
            'k': k,
            'run time': cal_time,
            'average cover size': cur_spread,
            'S': S
        })
    import pandas as pd

    df_IC_hep = pd.DataFrame(list_IC_hep)
    df_IC_hep.to_csv('../../data/output/degree/IC_DHICM(p=0.01)_hep_Graph.csv')
    print('文件输出完毕——结束')
