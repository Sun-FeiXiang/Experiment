"""
使用igraph
独立级联模型下，度折扣算法 degree discount heuristic [1]
[1] -- Wei Chen et al. Efficient influence maximization in Social Networks (algorithm 4)
"""
import time
from igraph import *
from diffusion.igraph_diffusion import IC
from heapdict import heapdict

def pageRank(G, k, p=.01):
    start_time = time.time()
    pages_rank = G.pagerank(eps=1e-4, damping=0.15)
    V = range(0, G.vcount())  # 节点索引值
    pr = heapdict()
    for u,p in zip(V,pages_rank):
        pr[u] = - p
    S, timelapse = [], []
    for _ in range(k):
        u,p = pr.popitem()
        S.append(u)
        timelapse.append(time.time() - start_time)
    return (S, timelapse)


if __name__ == "__main__":
    start = time.time()
    G = Graph.Read_Edgelist('../../data/graphdata/phy.txt', directed=False)
    read_time = time.time()
    print('读取网络时间：', read_time - start)
    p = 0.01
    output = pageRank(G, 50, p)
    list_IC_hep = []
    for k in range(1, 51):
        S = output[0][:k]
        cur_spread = IC(G, S, p, 1000)
        cal_time = output[1][k - 1]
        print('pagerank算法运行时间：', cal_time)
        print('k = ', k, '选取节点集为：', S)
        print('k=', k, '平均覆盖大小：', cur_spread)
        list_IC_hep.append({
            'k': k,
            'run time': cal_time,
            'average cover size': cur_spread,
            'S': S
        })
    import pandas as pd
    #
    # df_IC_hep = pd.DataFrame(list_IC_hep)
    # df_IC_hep.to_csv('../../data/output/pageRank/IC_pageRank(p=0.01)_phy_Graph.csv')
    # print('文件输出完毕——结束')
