"""
独立级联模型下，度折扣算法 degree discount heuristic [1]
A Novel Centrality Cascading Based Edge
Parameter Evaluation Method for Robust
Influence Maximization
2017 修改于度折扣 来源算法2
"""

from diffusion.Networkx_diffusion import spread_run_IC,spread_run_LT
from algorithm.priorityQueue import PriorityQueue as PQ  # priority queue
from timeit import default_timer as timer

def NewDiscount(G, k, p):

    S = []
    dd = PQ()
    t = dict()
    d = dict()

    for u in G.nodes():
        d[u] = sum([G[u][v]['weight'] for v in G[u]]) # each edge adds degree 1
        dd.add_task(u, -d[u]) # add degree of each node
        t[u] = 0

    for i in range(k):
        u, priority = dd.pop_item()
        S.append(u)
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight']
                priority = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p
                dd.add_task(v, -priority)
    return S

if __name__ == "__main__":
    import time

    start = time.time()
    from dataPreprocessing.read_txt_nx import read_Graph

    G = read_Graph('../../data/Amazon0302.txt')
    read_time = time.time()
    print('读取网络时间：', read_time - start)
    p = 0.05
    output = NewDiscount(G, 50, p)
    list_IC_hep = []
    for k in range(1, 51):
        S = output[0][:k]
        cur_spread = spread_run_IC(G, S, p, 10000)
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
    df_IC_hep.to_csv('../../data/output/degreeDiscount/IC_degreeDiscount(p=0.05)_ama_Graph.csv')
    print('文件输出完毕——结束')
