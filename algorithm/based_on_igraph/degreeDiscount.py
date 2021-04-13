"""
使用igraph
独立级联模型下，度折扣算法 degree discount heuristic [1]
[1] -- Wei Chen et al. Efficient influence maximization in Social Networks (algorithm 4)
"""
from heapdict import heapdict
from timeit import default_timer as timer
from igraph import *
from diffusion.igraph_diffusion import IC

def degreeDiscountIC(G, k, p=.01):
    """
    在独立级联模型中查找要传播的初始节点集（带优先级队列）
    Input: G -- igraph 图对象
    k -- 需要的节点个数
    p -- 传播概率
    Output:
    S -- 选择的k个点的集合
    """
    S, timelapse, start_time = [], [], timer()
    dd = heapdict()  # 度折扣
    t = dict()  #
    d = dict()  # 每个顶点的度

    V = range(0, G.vcount())  # 节点索引值
    # 初始度折扣
    for u in V:
        d[u] = sum([G[u, v] for v in G.neighbors(u)])
        dd[u] = -d[u]  # 添加每个节点的度数
        t[u] = 0

    # 贪心的给S加点
    for i in range(k):
        u, priority = dd.popitem() # 基于最大度折扣的节点提取 u及优先级代表节点及其度数
        S.append(u)
        timelapse.append(timer() - start_time)
        for v in G.neighbors(u):  # G[u]是u的邻接表
            if v not in S:
                t[v] += G[u, v]
                priority = d[v] - 2 * t[v] - (d[v] - t[v]) * t[v] * p
                dd[v] = -priority
    return (S, timelapse)


if __name__ == "__main__":
    import time

    start = time.time()
    G = Graph.Read_Edgelist('../../data/graphdata/hep.txt', directed=True)
    read_time = time.time()
    print('读取网络时间：', read_time - start)
    p = 0.01
    output = degreeDiscountIC(G, 50, p)
    list_IC_hep = []
    for k in range(1, 51):
        S = output[0][:k]
        cur_spread = IC(G, S, p, 100)
        cal_time = output[1][k - 1]
        print('degreeDiscount算法运行时间：', cal_time)
        print('k = ', k, '选取节点集为：', S)
        print('k=', k, '平均覆盖大小：', cur_spread)
        list_IC_hep.append({
            'k': k,
            'run time': cal_time,
            'average cover size': cur_spread,
            'S': S
        })
        temp_time = timer()  # 记录当前时间
    import pandas as pd

    df_IC_hep = pd.DataFrame(list_IC_hep)
    df_IC_hep.to_csv('../output/degreeDiscount/IC_degreeDiscount(p=0.01,igraph)_hep_Graph.csv')
    print('文件输出完毕——结束')
