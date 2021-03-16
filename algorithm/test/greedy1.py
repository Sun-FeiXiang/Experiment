"""
贪心算法：
1.计算每个节点被哪些节点影响(在每个活跃边图上使用bfs计算每个节点影响的节点集，重复多次，求出现频率高的)
2.当两个节点影响的个数相同时，优先选择其邻居影响总数最多的节点，选择出邻居的入邻居数大的节点并标记它所影响的节点集。
"""
import random
from copy import deepcopy
from timeit import default_timer as timer
import math
from diffusion.Networkx_diffusion import spread_run_IC

def findNR(G, Ep):
    # 从图G中移除阻塞边，获得传播图
    E = deepcopy(G)
    edge_rem = [e for e in E.edges() if random.random() < (1 - Ep[e]) ** (E[e[0]][e[1]]['weight'])]
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


def get_node_influence_set(G, Ep, R=100):
    """

    :param G:
    :param S: 种子集合
    :param R: 迭代次数
    :return:
    """
    node_frequency = dict()  # 记录节点出现的频率
    node_influence_num = dict()  # 记录节点影响的平均个数
    for _ in range(R):
        influence_set = findNR(G, Ep)  # 影响的节点集合
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


def get_best_value_node(node_influence_set):
    """
    按照影响个数排序
    :param node_influence_set:
    :return:
    """
    node_influence_set = sorted(node_influence_set.items(), key=lambda x: len(x[1]), reverse=True)
    return node_influence_set[0][0]


def update_influence_set(node_influence_set, node):
    """
    更新节点影响集
    :param node_influence_set:
    :param node:
    :return:
    """
    influence_set = [node]  # 影响节点集合
    influence_set.extend(node_influence_set[node])
    for influence_node in influence_set:
        neighbors = G.neighbors(influence_node)
        for neighbor in neighbors:
            if neighbor in node_influence_set.keys():
                node_influence_set[neighbor] = list(
                    set(node_influence_set[neighbor]).difference(set(influence_set)))  # 作差集
    node_influence_set.pop(node)
    return node_influence_set


def greedy(G, k, Ep, mc=1000):
    node_influence_set = get_node_influence_set(G, Ep)  # 影响的节点集

    S,timelapse,spread = [],[],[]
    start_time = timer()
    for i in range(k):
        u = get_best_value_node(node_influence_set)
        S.append(u)
        node_influence_set = update_influence_set(node_influence_set, u)  # 更新影响的节点集
        timelapse.append(timer()-start_time)
    ss = []
    for s in S:
        ss.append(s)
        cur_spread = spread_run_IC(G,ss,0.01,mc)
        spread.append(cur_spread)
    return (S,spread,timelapse)


if __name__ == "__main__":
    import time

    start = time.time()
    from dataPreprocessing.read_txt_nx import read_Graph

    G = read_Graph("../../data/graphdata/hep.txt")
    read_time = time.time()
    print('读取网络时间：', read_time - start)
    from dataPreprocessing.generation_propagation_probability import fixed_probability

    Ep = fixed_probability(G, 0.01)
    # node_influence_set = get_node_influence_set(G, Ep, 100)
    output = greedy(G, 50, Ep, 1000)

    list_IC_hep = []
    for k in range(1, 51):
        S = output[0][:k]
        cur_spread = output[1][k - 1]
        cal_time = output[2][k - 1]
        print('greedy算法运行时间：', cal_time)
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
    df_IC_hep.to_csv('../../data/output/greedy/IC_greedy1_hep.csv')
    print('文件输出完毕——结束')
