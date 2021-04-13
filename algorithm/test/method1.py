"""
omega = sum(node.degree())
local influence = alpha * math.sqrt(k_truss^2+d^2)
更新
"""
import random
from algorithm.basedCore.k_truss import k_truss
from timeit import default_timer as timer
import math
from copy import deepcopy
from heapdict import heapdict
from diffusion.Networkx_diffusion import spread_run_IC


def get_node_CC(G, S, node):
    """
    获得节点的两层，及各层拥有的种子个数
    :param G:
    :param node:
    :param p:
    :return:
    """
    C1 = []
    C2 = []
    q = []  # 队列
    q.append(node)
    level = 0  # 覆盖第几层
    while len(q) > 0 and level < 2:
        v = q.pop(0)  # 弹出第一个节点
        if level == 0:
            C1.append(v)
        else:
            C2.append(v)
        neighbors = list(G.neighbors(v))
        for u in neighbors:
            if u not in C1 and u not in C2:
                q.append(u)
        level = level + 1  # 访问一层
    C1_S = [x for x in C1 if C1 in S]
    C2_S = [y for y in C2 if C2 in S]
    return [len(C1), len(C2), len(C1_S), len(C2_S)]


def get_node_DD(G, node, p):
    """
    获取节点的传播度
    :param G:
    :param node:
    :param p:
    :return:
    """
    C_DD_pp = 0  # 节点v邻居的贡献
    C_DD_p = 0  # 节点v的贡献
    u = list(G.neighbors(node))
    for u_i in u:
        C_DD_p = C_DD_p + G[node][u_i]['weight']
        w = list(G.neighbors(u_i))
        w.remove(node)
        C_D_i = 0
        for w_i in w:
            C_D_i = C_D_i + G[u_i][w_i]['weight']
        C_DD_pp = C_DD_pp + C_D_i
    return (C_DD_p + C_DD_pp) * p


def findNR(G, p):
    # 从图G中移除阻塞边，获得传播图
    E = deepcopy(G)
    edge_rem = [e for e in E.edges() if random.random() < (1 - p) ** (E[e[0]][e[1]]['weight'])]
    E.remove_edges_from(edge_rem)
    # 初始化 CC
    CCs = dict()  # each component is reflection of the number of a component to its members
    explored = dict(zip(E.nodes(), [False] * len(E)))
    c = 0
    # perform BFS to discover CC
    for node in E:
        if not explored[node]:
            c += 1
            explored[node] = True
            CCs[c] = [node]
            component = list(E[node].keys())
            for neighbor in component:
                if not explored[neighbor]:
                    explored[neighbor] = True
                    CCs[c].append(neighbor)
                    component.extend(E[neighbor].keys())
    # 转换每块Rancas为每个节点能到达的集合 node reach
    NR = dict()
    for line_value in CCs.values():
        if len(line_value) == 0:
            NR[line_value[0]] = []
        else:
            for one in line_value:
                a = line_value.copy()
                a.remove(one)
                NR[one] = a
    return NR


def get_node_influence_set(G, p, R=100):
    """

    :param G:
    :param S: 种子集合
    :param R: 迭代次数
    :return:
    """
    node_frequency = dict()  # 记录节点出现的频率
    node_influence_num = dict()  # 记录节点影响的平均个数
    for _ in range(R):
        influence_set = findNR(G, p)  # 影响的节点集合
        for key, value in influence_set.items():
            if key in node_frequency.keys():
                node_frequency_line = node_frequency[key]
                for one_value in value:
                    if one_value in node_frequency_line.keys():
                        node_frequency_line[one_value] = node_frequency_line[one_value] + 1
                    else:
                        node_frequency_line[one_value] = 1
                node_frequency[key] = node_frequency_line
            else:
                node_frequency_line = dict()
                for one_value in value:
                    node_frequency_line[one_value] = 1
                node_frequency[key] = node_frequency_line
            if key in node_influence_num.keys():
                node_influence_num[key] = node_influence_num[key] + len(value) / R
            else:
                node_influence_num[key] = len(value) / R

    node_influence_set = dict()
    # 根据节点出现频率和平均影响个数得出影响节点集
    for node, num in node_influence_num.items():
        num = math.ceil(num)
        cur_node_frequency = node_frequency[node]
        cur_node_frequency = sorted(cur_node_frequency.items(), key=lambda x: x[1], reverse=True)
        if len(cur_node_frequency) == 0 or num == 0:
            node_influence_set[node] = []
        else:
            cur_set = []
            for line in cur_node_frequency:
                cur_set.append(line[0])
                if len(cur_set) == num:
                    break
            node_influence_set[node] = cur_set
    return node_influence_set


def method(G, k, p, mc=1000):
    node_influence_set = get_node_influence_set(G, p)  # 影响的节点集
    All_DD = heapdict()
    for node in G.nodes:
        All_DD[node] = -get_node_DD(G,node,p)
    S, timelapse, spread = [], [], []
    start_time = timer()
    S_p = [] # 已经覆盖的节点
    while len(S) < k:
        node,DD = All_DD.popitem()
        if node not in S:
            S.append(node)
            cur_spread = spread_run_IC(G, S, 0.01, mc)
            spread.append(cur_spread)
            S_p.append(node)
            S_p.extend(node_influence_set[node])
            timelapse.append(timer() - start_time)

    return (S, spread, timelapse)

if __name__ == "__main__":
    import time

    start = time.time()
    from dataPreprocessing.read_txt_nx import read_Graph

    G = read_Graph("../../data/graphdata/hep.txt")
    read_time = time.time()
    print('读取网络时间：', read_time - start)
    diffusionDegree_output = method(G, 50, 0.01, 100)

    list_IC_hep = []
    for k in range(1, 51):
        S = diffusionDegree_output[0][:k]
        cur_spread = diffusionDegree_output[1][k - 1]
        cal_time = diffusionDegree_output[2][k - 1]
        print('method算法运行时间：', cal_time)
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
    df_IC_hep.to_csv('../../data/output/test/IC_method_hep_Graph.csv')
    print('文件输出完毕——结束')
