"""
Identification of multi-spreader users in social networks for viral marketing


"""
from heapdict import heapdict
from timeit import default_timer as timer
from diffusion.Networkx_diffusion import spread_run_IC, spread_run_LT
import math
from dataPreprocessing.read_txt_nx import read_Graph, avg_degree
import time
from dataPreprocessing.generation_propagation_probability import fixed_probability


def get_node_core(g):
    """
    修改的，求节点的核心值
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
            # print(sorted(node_degree.items(),key=lambda x: x[1]))
            if min(node_degree.items(), key=lambda x: x[1])[1] > level:
                break

        level = min(node_degree.items(), key=lambda x: x[1])[1]
    return k_nodes


def get_node_degree(G):
    """
    获取节点的度（两个节点之间至少有一条边）
    :param G:
    :return:节点的度
    """
    d = dict()
    for u in G.nodes:
        d[u] = sum([G[u][v]['weight'] for v in G[u]])
    return d


def get_node_entropy(G, node_core):
    node_entropy = dict()
    for node in G.nodes:
        cur_entropy = 0
        neighbors = list(G.neighbors(node))
        neighbors_core = dict()  # 邻居在每个核心的个数core:num
        for u in neighbors:
            cur_node_core = node_core[u]  # 获取当前节点的核心值
            if cur_node_core in neighbors_core.keys():
                neighbors_core[cur_node_core] = neighbors_core[cur_node_core] + 1
            else:
                neighbors_core[cur_node_core] = 1
        for core, num in neighbors_core.items():
            p_i = num / len(neighbors)
            cur_entropy = cur_entropy - p_i * math.log2(p_i)
        node_entropy[node] = cur_entropy
    return node_entropy


def MCDE(G, k, Ep, theta, eta, alpha, beta, gamma):
    start_time = timer()
    node_degree = get_node_degree(G)
    node_core = get_node_core(G)
    node_entropy = get_node_entropy(G, node_core)
    mcde = heapdict()
    for u in G.nodes:
        mcde[u] = - (alpha * node_core[u] + beta * node_degree[u] + gamma * node_entropy[u])
    S, timelapse = [], []
    while len(S) < k:
        u, u_pn = mcde.popitem()
        sel = True
        for v in S:
            if u in G.neighbors(v):
                if Embeddeness(G, u, v) > theta:#  or Ep[(u, v)] > eta
                    sel = False
        if sel:
            S.append(u)
            timelapse.append(timer() - start_time)
    return (S, timelapse)


def Embeddeness(G, A, B):
    A_neighbors = set(G.neighbors(A))
    B_neighbors = set(G.neighbors(B))
    A_B_intersection = A_neighbors.intersection(B_neighbors)
    # A_B_union = A_neighbors.union(B_neighbors)
    return len(A_B_intersection)


if __name__ == "__main__":
    start = time.time()
    G = read_Graph("../data/graphdata/hep.txt")
    read_time = time.time()
    print('读取网络时间：', read_time - start)
    p = 0.01
    Ep = fixed_probability(G, p)
    theta = avg_degree(G)
    algorithm_output = MCDE(G, 50, p, theta*2, Ep, 1, 1, 1)
    list_IC_hep = []
    for k in range(1, 51):
        S = algorithm_output[0][:k]
        cur_spread = spread_run_IC(G, S, p, 1000)
        cal_time = algorithm_output[1][k - 1]
        print('MCDE算法运行时间：', cal_time)
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
    df_IC_hep.to_csv('../data/output/test/IC_MCDE(p=0.01)_hep_Graph.csv')
    print('文件输出完毕——结束')
