"""
来源：Efficient Influence Maximization in Social-Networks Under Independent Cascade Model

"""
from diffusion.Networkx_diffusion import spread_run_IC
from timeit import default_timer as timer


def DHICM(G, k, p, mc=100):
    d = dict()
    dd = dict()  # 度折扣
    t = dict()  # 选择邻居的个数
    S, spread, timelapse, start_time = [], [], [], timer()
    for u in G:
        d[u] = sum([G[u][v]['weight'] for v in G[u]])  # 每条边增加的度 一般是1，也有两个点之间有多条边
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd[u] = d[u]
        t[u] = 0
    for i in range(k):
        u, ddv = max(dd.items())
        dd.pop(u)
        S.append(u)
        cur_spread = spread_run_IC(G, S, p, mc)
        spread.append(cur_spread)
        timelapse.append(timer() - start_time)
        for v in G[u]:
            if v not in S:
                dd[v] = d[v] - 1 - (d[u] - 1) * p
    return S, spread, timelapse


if __name__ == "__main__":
    import time

    start = time.time()
    from dataPreprocessing.read_txt_nx import read_Graph

    G = read_Graph('../../data/graphdata/hep.txt')
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    output = DHICM(G, 50, 0.01, 1000)

    list_IC_hep = []
    for k in range(1, 51):
        S = output[0][:k]
        cur_spread = output[1][k - 1]
        cal_time = output[2][k - 1]
        print('DHICM算法运行时间：', cal_time)
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
    df_IC_hep.to_csv('../../data/output/degree/IC_DHICM_hep_Graph.csv')
    print('文件输出完毕——结束')

