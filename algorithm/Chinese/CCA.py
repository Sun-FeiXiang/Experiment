from model.ICM_nx import spread_run_IC,IC
import time
from preprocessing.read_txt_nx import read_Graph
from preprocessing.generation_propagation_probability import p_fixed,p_random,p_inEdge,fixed_weight,p_fixed_with_link
import networkx as nx


def get_node_degree(G):
    """
    获取节点的度（两个节点之间至少有一条边）
    :param G:
    :return:节点的度
    """
    d = dict()
    for u in G.nodes:
        #d[u] = sum([G[u][v]['weight'] for v in G[u]])
        d[u] = len(G[u])
    return d


def node_core_number(g):
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


def mark_overlay(G, node, CO_v, d):
    """
    使用bfs覆盖
    :param G: networkx对象
    :param node: 开始节点
    :param d: 度
    :param CO_v:访问标识
    :return: 覆盖的节点集
    """
    cover_nodes = [node]
    q = []  # 队列
    q.append(node)
    CO_v[node] = True
    level = 0  # 覆盖第几层
    while len(q) > 0 and level < d:
        v = q.pop(0)  # 弹出第一个节点
        G_adj = G.adj[v]
        for key, value in G_adj.items():
            if not CO_v[key]:
                CO_v[key] = True
                cover_nodes.append(key)
                q.append(key)
        level = level + 1  # 访问一层
    return cover_nodes


def get_max_core_num(node_core):
    return sorted(node_core.items(), key=lambda x: x[1], reverse=True)[0][1]


def CCA(G, k, d):
    """
    :param G: networkx图对象
    :param k: 初始节点集的节点个数
    :param d: 将距离为d的节点标记为覆盖
    :return: 选择的k个点的集合
    """
    start_time = time.time()
    S, timelapse = [], []
    node_degree = get_node_degree(G)
    node_core = node_core_number(G)
    CO_v = dict()  # 节点覆盖属性
    for node in G.nodes:
        CO_v[node] = False
    for _ in range(k):
        max_core_node_list = []  # 最大核列表
        max_core = get_max_core_num(node_core)
        for node, core in node_core.items():
            if core == max_core and not CO_v[node]:
                max_core_node_list.append(node)
        D = dict()  # 拥有
        for u in max_core_node_list:
            if not CO_v[u] and u in node_degree.keys():
                D[u] = node_degree[u]
        # print(D)
        seed = sorted(D.items(), key=lambda x: x[1], reverse=True)[0][0]
        S.append(seed)
        cover_nodes = mark_overlay(G, seed, CO_v, d)
        for cover_node in cover_nodes:
            del node_core[cover_node]
        timelapse.append(time.time() - start_time)

    return (S, timelapse)


if __name__ == "__main__":
    start = time.time()
    G = read_Graph("../../data/graphdata/phy.txt")
    # G = nx.read_edgelist("../../data/graphdata/PGP.txt", nodetype=int,create_using=nx.Graph)  # 其他数据集使用此方式读取
    # fixed_weight(G)
    read_time = time.time()
    print('读取网络时间：', read_time - start)
    p = 0.05
    I = 1000
    p_fixed_with_link(G,p)
    # p_fixed(G,p)
    # p_inEdge(G)
    d = 2
    algorithm_output = CCA(G, 50, d)
    list_IC_hep = []
    print("p=",p,"R,I=",I,",data=phy,Graph")
    for k in range(1, 51):
        S = algorithm_output[0][:k]
        cur_spread = IC(G, S, I)
        cal_time = algorithm_output[1][k - 1]
        print('CCA算法运行时间：', cal_time)
        print('k = ', k, '选取节点集为：', S)
        print('k=', k, '平均覆盖大小：', cur_spread)
        list_IC_hep.append({
            'k': k,
            'run time': cal_time,
            'average cover size': cur_spread,
            'S': S
        })
    import pandas as pd

    df_IC_hep = pd.DataFrame(list_IC_hep)
    df_IC_hep.to_csv('../../data/output/CCA/IC_CCA2(p=0.05R,I=1000)_phy_Graph.csv')
    print('文件输出完毕——结束')
