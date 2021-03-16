"""
perfect node Discount
PN=d^2+k_truss^2
初始网络k_truss值不变
折扣为，当v有其n邻居被选为种子，其中他们有m条边时，PN=PN-2dm+m^2
"""
from algorithm.K_core.k_truss import k_truss
from heapdict import heapdict
from timeit import default_timer as timer
from diffusion.Networkx_diffusion import spread_run_IC


def get_node_degree(G):
    """
    获取节点的度，两个节点之间至少有一条边
    :param G:
    :return:
    """
    d = dict()
    for u in G.nodes:
        d[u] = sum([G[u][v]['weight'] for v in G[u]])
    return d


def PN_Discount(G, k, p, mc=1000):
    node_k_truss = k_truss(G)
    node_degree = get_node_degree(G)
    pn = heapdict()
    t = dict()
    for u in G.nodes:
        pn[u] = -(node_degree[u] ** 2 + node_k_truss[u] ** 2)
        t[u] = 0

    S, timelapse, spread = [], [], []
    start_time = timer()
    for _ in range(k):
        u, u_pn = pn.popitem()
        S.append(u)
        cur_spread = spread_run_IC(G, S, p, mc)
        spread.append(cur_spread)
        timelapse.append(timer() - start_time)
        for v in G[u]:  # G[u]是u的邻接表
            if v not in S:  # ！！！
                t[v] += G[u][v]['weight']
                u_pn = u_pn + 2 * t[v] * node_degree[v] - t[v] ** 2
                pn[v] = u_pn
    return (S, spread, timelapse)


if __name__ == "__main__":
    import time

    start = time.time()
    from dataPreprocessing.read_txt_nx import read_Graph

    G = read_Graph("../../data/graphdata/hep.txt")
    read_time = time.time()
    print('读取网络时间：', read_time - start)
    PN_Discount_output = PN_Discount(G, 50, 0.01, 100)

    list_IC_hep = []
    for k in range(1, 51):
        S = PN_Discount_output[0][:k]
        cur_spread = PN_Discount_output[1][k - 1]
        cal_time = PN_Discount_output[2][k - 1]
        print('PN discount算法运行时间：', cal_time)
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
    df_IC_hep.to_csv('../../data/output/test/IC_PN_Discount_hep_Graph.csv')
    print('文件输出完毕——结束')
