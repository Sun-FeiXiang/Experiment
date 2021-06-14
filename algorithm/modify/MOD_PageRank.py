import networkx as nx
from model.ICM_nx import spread_run_IC,IC
from preprocessing.read_txt_nx import read_Graph
from preprocessing.generation_propagation_probability import p_fixed,p_random,fixed_weight,p_inEdge,p_fixed_with_link
from time import time
from heapdict import heapdict
"""
PageRank作为一个经典的网页排序算法，在影响最大化中的应用，一般作为一个对照实验。本处的实现主要依据下面论文中的设置
来源：Scalable Influence Maximization for Prevalent Viral Marketing in Large-Scale Social Networks∗
对比实验
"""


def path_cover(G, CO_v, node, edge_truss_number, c):
    """
    层次遍历，优先选择truss值大的边进行覆盖
    :param G:
    :param CO_v:覆盖标识集
    :param node:节点
    :param edge_truss_number:边的turss值
    :param c: 每层的覆盖率
    :return: 覆盖到的点集
    """
    q = [node]
    node_neighbors_num = len(list(G.neighbors(node)))  # 节点的邻居个数设置为该节点的覆盖大小
    cover_list = []
    while len(q) > 0 and len(cover_list) < node_neighbors_num:
        u = q.pop(0)
        u_neighbors = list(G.neighbors(u))
        if len(u_neighbors) == 0:  # 如果该点没有邻居则继续
            continue
        cover_num = round(c * len(u_neighbors))  # 覆盖个数等于覆盖概率乘以邻居个数 四舍五入取整
        if cover_num == 0:
            cover_num = 1
        adj_truss_number = dict()  # 邻边的truss值
        for v in u_neighbors:
            adj_truss_number[u, v] = edge_truss_number[u, v]
        adj_truss_number = sorted(adj_truss_number.items(), key=lambda x: x[1], reverse=True)
        u_cover_list = []
        CO_v[u] = True
        i,j = 0,0
        while i < cover_num and j < len(adj_truss_number):  # 选择未被覆盖的前几个,i不能无限制的加
            top_truss = adj_truss_number[j][0][1]
            if not CO_v[top_truss]:
                CO_v[top_truss] = True
                u_cover_list.append(top_truss)
                i = i + 1
            j = j + 1
        cover_list.extend(u_cover_list)  # 总覆盖的节点
        q.extend(u_cover_list)
    choose = True  # 是否选择该节点作为种子节点
    if len(cover_list) < node_neighbors_num:
        choose = False  # 覆盖不够，不选为种子节点
    return choose, cover_list


def get_edge_truss_number(G):
    """
    有修改的，求边的truss值
    :param G:
    :return:节点的truss值
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
        truss_number[edge[1], edge[0]] = truss_number[edge]  # 无向图
    return truss_number


def pageRank(G, k,p):
    start_time = time()
    pages_rank = nx.pagerank(G, tol=1e-4,alpha=0.15,weight='weight')
    pages_rank = sorted(pages_rank.items(), key=lambda x: x[1], reverse=True)
    edge_truss_num = get_edge_truss_number(G)
    S, timelapse= [], []
    pn = heapdict()
    for u, ranking in pages_rank:
        pn[u] = ranking
    CO_v = dict()
    for u in G.nodes:
        CO_v[u] = False
    c = p*10
    i = 0
    while i < k:
        u, u_pn = pn.popitem()
        timelapse.append(time() - start_time)
        is_choose, cur_cover_list = path_cover(G, CO_v, u, edge_truss_num, c)
        # print(is_choose)
        cur_cover_list.append(u)  # 当前节点覆盖的节点集,加入u
        if is_choose:  # 覆盖结构足够d个，选为种子节点
            i = i + 1
            S.append(u)
            for cover_one in cur_cover_list:  # 弹出这些节点
                if cover_one in pn.keys():
                    pn.pop(cover_one)
        else:  # 取消此部分的覆盖标识
            for cover_one in cur_cover_list:
                CO_v[cover_one] = False
    return (S, timelapse)


if __name__ == "__main__":
    start = time()
    G = read_Graph("../../data/graphdata/hep.txt")
    # G = nx.read_edgelist("../../data/graphdata/email.txt", nodetype=int,create_using=nx.Graph)  # 其他数据集使用此方式读取
    # fixed_weight(G)
    read_time = time()
    print('读取网络时间：', read_time - start)
    p = 0.01
    I = 1000
    p_fixed(G,p)
    # p_inEdge(G)
    # p_fixed_with_link(G,p)
    algorithm_output = pageRank(G, 50,p)
    print("p=",p,",I=",I,"data=email,Graph")
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

    # df_IC_random_hep = pd.DataFrame(list_IC_hep)
    # df_IC_random_hep.to_csv('../data/output/pageRank/IC_pageRank(p=0.01,I=1000)_DBLP_Graph.csv')
    # print('文件输出完毕——结束')
