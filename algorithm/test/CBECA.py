"""
Core-based edge covering algorithm
基于核的边覆盖算法
覆盖系数 c:(0,1)
优先选择pn=sqrt(d**2+k_s**2)大的节点
然后利用k-truss，计算边的truss值
选择pn值大的点，覆盖周围truss值大的边，并标记相应的点，更新周围pn值

"""
from algorithm.K_core.k_truss import k_truss
from heapdict import heapdict
from timeit import default_timer as timer
from diffusion.Networkx_diffusion import spread_run_IC

def edge_truss_number(G):
    """

    :param G:
    :return:
    """
    truss_number = dict()
    for edge in G.edges:  # O(m*k^2)
        start = edge[0]
        end = edge[1]
        start_adj = list(G.adj[start].keys())
        end_adj = list(G.adj[end].keys())
        intersection = [i for i in start_adj if i in end_adj]  # O(k^2)
        inter_node_num = 0
        for inter in intersection:
            inter_node_num = inter_node_num + min(G[start][inter]['weight'], G[end][inter]['weight'])
        truss_number[edge] = inter_node_num + 1 + G[start][end]['weight']  # 边的truss值是交集+2+两点之间边的个数-1
    return truss_number


def node_core_number(g):
    """
    :param G:
    :return: 所有节点的核心值
    """
    G = g.copy()
    k_nodes = dict()
    level = 1
    node_degree = get_node_degree(G)
    while len(node_degree):
        while True:
            level_node_list = []
            for item in node_degree.items():  # 返回节点及其度
                if item[1] <= level:
                    level_node_list.append(item[0])
                    # 这里设置了value是从1开始的；
                    k_nodes[item[0]] = level
            G.remove_nodes_from(level_node_list)
            node_degree = get_node_degree(G)
            if not len(node_degree):
                return k_nodes
            #print(sorted(node_degree.items(),key=lambda x: x[1]))
            if min(node_degree.items(), key=lambda x: x[1])[1] > level:
                break

        level = min(node_degree.items(), key=lambda x: x[1])[1]
    return k_nodes


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


def edge_cover(G,edge_truss_number,c):

    # 层次遍历
    b 

def CBECA(G, k, p, mc=1000,c=0.2):
    edge_truss_num = edge_truss_number(G)
    node_core_num = node_core_number(G)
    node_degree = get_node_degree(G)

    pn = heapdict()
    t = dict()
    for u in G.nodes:
        pn[u] = -(node_degree[u] ** 2 + node_core_num[u] ** 2)
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


if __name__=="__main__":
    import time

    start = time.time()
    from dataPreprocessing.read_txt_nx import read_Graph

    G = read_Graph("../../data/graphdata/hep.txt")
    read_time = time.time()
    print('读取网络时间：', read_time - start)
    PN_Discount_output = CBECA(G, 50, 0.01, 100)

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
    df_IC_hep.to_csv('../../data/output/test/IC_PN_Discount(core)_hep_Graph.csv')
    print('文件输出完毕——结束')
