"""
Identification of multi-spreader users in social networks for viral marketing
原论文中，边的权重设为（0，1）的随机值
theta = 网络的平均度

"""
import networkx as nx
from heapdict import heapdict
from timeit import default_timer as timer
from model.ICM_nx import spread_run_IC, IC
import math
from preprocessing.read_txt_nx import read_Graph, avg_degree, avg_degree2
import time
from preprocessing.generation_propagation_probability import fixed_probability, p_random, p_fixed,p_inEdge,p_fixed_with_link,fixed_weight
from preprocessing.generation_node_threshold import random_threshold


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
        d[u] = G.degree[u]#sum([G[u][v]['weight'] for v in G[u]])  #
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
            cur_entropy = cur_entropy + p_i * math.log2(p_i)
        node_entropy[node] = -cur_entropy
    return node_entropy



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



def MCDE(G, k,p, theta, node_threshold, alpha, beta, gamma):
    start_time = timer()
    node_degree = get_node_degree(G)
    node_core = get_node_core(G)
    node_entropy = get_node_entropy(G, node_core)
    edge_truss_num = get_edge_truss_number(G)
    mcde = heapdict()
    CO_v = dict()
    for u in G.nodes:
        mcde[u] = - (alpha * node_core[u] + beta * node_degree[u] + gamma * node_entropy[u])
        CO_v[u] = False
    S, timelapse = [], []
    c = p*10
    i = 0
    while i < k:
        u, u_pn = mcde.popitem()
        timelapse.append(timer() - start_time)
        is_choose, cur_cover_list = path_cover(G, CO_v, u, edge_truss_num, c)
        # print(is_choose)
        cur_cover_list.append(u)  # 当前节点覆盖的节点集,加入u
        if is_choose:  # 覆盖结构足够d个，选为种子节点
            i = i + 1
            S.append(u)
            for cover_one in cur_cover_list:  # 弹出这些节点
                if cover_one in mcde.keys():
                    mcde.pop(cover_one)
        else:  # 取消此部分的覆盖标识
            for cover_one in cur_cover_list:
                CO_v[cover_one] = False
    return (S, timelapse)


def Embeddeness(G, A, B):
    A_neighbors = set(G.neighbors(A))
    B_neighbors = set(G.neighbors(B))
    A_B_intersection = A_neighbors.intersection(B_neighbors)
    # A_B_union = A_neighbors.union(B_neighbors)
    return len(A_B_intersection)


if __name__ == "__main__":
    start = time.time()
    G = read_Graph("../../data/graphdata/hep.txt")  # 针对hep和phy数据集使用该函数读取网络
    # G = nx.read_edgelist("../data/graphdata/facebook_combined.txt",nodetype=int,create_using=nx.Graph) #其他数据集使用此方式读取
    # fixed_weight(G)
    read_time = time.time()
    print('读取网络时间：', read_time - start)
    p = 0.01
    I = 1000
    p_fixed(G, p)
    # p_fixed_with_link(G,p)
    theta = avg_degree(G)  # 平均度,设置了weight之后都用这个
    node_threshold = random_threshold(G)  # 节点设置阈值为（0，1）的随机数
    algorithm_output = MCDE(G, 50,p, theta, node_threshold, 1, 1, 1)
    list_IC_hep = []
    print("p=",p,"R,I=",I,",data=hep,Graph")
    for k in range(1, 51):
        S = algorithm_output[0][:k]
        cur_spread = IC(G, S, I)
        cal_time = algorithm_output[1][k - 1]
        print('MOD_MCDE算法运行时间：', cal_time)
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

    # df_IC_hep = pd.DataFrame(list_IC_hep)
    # df_IC_hep.to_csv('../data/output/MCDE/IC_MCDE(p=0.01,I=1000)_hep_Graph.csv')
    # print('文件输出完毕——结束')
