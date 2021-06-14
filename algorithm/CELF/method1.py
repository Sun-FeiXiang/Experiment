from model.ICM_nx import spread_run_IC, IC
import math
import networkx as nx
from preprocessing.generation_propagation_probability import p_fixed, fixed_weight, p_random, p_inEdge, \
    p_fixed_with_link
from time import time
from preprocessing.read_txt_nx import read_Graph, avg_degree
import pandas as pd

def get_node_Influence(G,S,p,node):
    """
    获得node被影响的概率
    :param G:
    :param S: 种子集
    :param p: 概率
    :param node: 当前种子
    :return:
    """
    t = 0 # 和邻居中种子的边数
    for v in list(G.neighbors(node)):
        if v in S:
            t = t + G[node][v]['weight']
    return 1-(1-p)**t

def get_S_Influence(G,S,p):
    """
    获得S的影响
    :param G: 图
    :param S: 种子集
    :param p: 概率
    :return:
    """
    N_S = []
    for u in S:
        N_S.extend(list(G.neighbors(u)))
    N_S = list(set(N_S))
    g_S = len(S)
    for v in N_S:
        g_S = g_S + get_node_Influence(G,S,p,v)
    return g_S

def CELF(G, k, p=0.01):
    start_time = time()
    S,timelapse = [],[]
    for i in range(k):
        max_g_S = 0
        v = -1
        for u in G.nodes:
            if u not in S:
                cur_g_S = get_S_Influence(G,S+[u],p)
                if cur_g_S > max_g_S:
                    max_g_S = cur_g_S
                    v = u
        if v != -1:
            S.append(v)
            timelapse.append(time()-start_time)
    return (S,timelapse)


if __name__ == "__main__":
    start = time()
    G = read_Graph("../../data/graphdata/hep.txt", directed=False)
    # G = nx.read_edgelist("../../data/graphdata/PGP.txt", nodetype=int, create_using=nx.Graph)  # 其他数据集使用此方式读取
    # fixed_weight(G)
    read_time = time()
    print('读取网络时间：', read_time - start)
    p = 0.05
    I = 1000
    # p_fixed_with_link(G, p)
    p_fixed(G, p)
    # p_random(G)
    # p_inEdge(G)
    algorithm_output = CELF(G, 50,p)
    list_IC_hep = []
    print("p=", p, ",I=", I, ",data=hep,Graph")
    for k in range(1, 51):
        S = algorithm_output[0][:k]
        cur_spread = IC(G, S, I)
        cal_time = algorithm_output[1][k - 1]
        print('method1算法运行时间：', cal_time)
        print('k = ', k, '选取节点集为：', S)
        print('k=', k, '平均覆盖大小：', cur_spread)
        list_IC_hep.append({
            'k': k,
            'run time': cal_time,
            'average cover size': cur_spread,
            'S': S
        })
    df_IC_hep = pd.DataFrame(list_IC_hep)
    df_IC_hep.to_csv('../../data/output/test/IC_newMethod(p=0.05,I=1000)_hep_Graph.csv')
    print('文件输出完毕——结束')


