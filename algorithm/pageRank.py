import networkx as nx
from model.ICM_nx import spread_run_IC,IC
from preprocessing.read_txt_nx import read_Graph
from preprocessing.generation_propagation_probability import p_fixed,p_random,fixed_weight,p_inEdge,p_fixed_with_link
from time import time
import numpy as np
from scipy.sparse import csc_matrix

"""
PageRank作为一个经典的网页排序算法，在影响最大化中的应用，一般作为一个对照实验。本处的实现主要依据下面论文中的设置
来源：Scalable Influence Maximization for Prevalent Viral Marketing in Large-Scale Social Networks∗
对比实验
"""

def pageRank(G, k):
    start_time = time()
    pages_rank = nx.pagerank(G, tol=1e-4,alpha=0.15,weight='weight')
    pages_rank = sorted(pages_rank.items(), key=lambda x: x[1], reverse=True)
    S, timelapse= [], []
    for u, ranking in pages_rank:
        if len(S) == k:
            break
        S.append(u)
        timelapse.append(time() - start_time)
    return (S, timelapse)



if __name__ == "__main__":
    start = time()
    # G = read_Graph("../data/graphdata/hep.txt")
    G = nx.read_edgelist("../data/graphdata/DBLP.txt", nodetype=int,create_using=nx.Graph)  # 其他数据集使用此方式读取
    fixed_weight(G)
    read_time = time()
    print('读取网络时间：', read_time - start)
    p = 0.01
    I = 1000
    p_fixed(G,p)
    # p_inEdge(G)
    # p_fixed_with_link(G,p)
    algorithm_output = pageRank(G, 50)
    print("p=",p,",I=",I,"data=DBLP,Graph")
    list_IC_hep = []
    for k in range(1, 51):
        S = algorithm_output[0][:k]
        cur_spread = IC(G,S,I)
        cal_time = algorithm_output[1][k - 1]
        print('pageRank算法运行时间：', cal_time)
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
    df_IC_random_hep.to_csv('../data/output/pageRank/IC_pageRank(p=0.01,I=1000)_DBLP_Graph.csv')
    print('文件输出完毕——结束')
