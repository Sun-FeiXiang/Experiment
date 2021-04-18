"""
论文：A New Centrality Measure for Influence Maximization in Social Networks
2011年 基于扩散度的启发式算法
"""

from heapdict import heapdict
from timeit import default_timer as timer
from diffusion.Networkx_diffusion import spread_run_IC


def diffusionDegree(G, k, p, mc=100):
    S, spread, timelapse = [], [], []
    start_time = timer()
    Q = heapdict()
    for v in G.nodes:
        C_DD_pp = 0  # 节点v邻居的贡献
        C_DD_p = 0  # 节点v的贡献
        u = list(G.neighbors(v))
        for u_i in u:
            C_DD_p = C_DD_p + G[v][u_i]['weight']
            w = list(G.neighbors(u_i))
            w.remove(v)
            C_D_i = 0
            for w_i in w:
                C_D_i = C_D_i + G[u_i][w_i]['weight']
            C_DD_pp = C_DD_pp + C_D_i * p
        Q[v] = -(C_DD_p*p + C_DD_pp)
    for _ in range(k):
        v, value = Q.peekitem()
        del Q[v]
        S.append(v)
        timelapse.append(timer() - start_time)
        spread.append(spread_run_IC(G, S, p, mc))
    return (S, spread, timelapse)


if __name__ == "__main__":
    import time

    start = time.time()
    from dataPreprocessing.read_txt_nx import read_Graph

    G = read_Graph("../../data/graphdata/DBLP.txt")
    read_time = time.time()
    print('读取网络时间：', read_time - start)
    diffusionDegree_output = diffusionDegree(G, 50, 0.05, 1000)

    list_IC_hep = []
    for k in range(1, 51):
        S = diffusionDegree_output[0][:k]
        cur_spread = diffusionDegree_output[1][k - 1]
        cal_time = diffusionDegree_output[2][k - 1]
        print('diffusionDegree算法运行时间：', cal_time)
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
    df_IC_hep.to_csv('../data/output/degree/IC_diffusionDegree_DBLP_Graph.csv')
    print('文件输出完毕——结束')
