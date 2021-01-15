"""
使用networkx网络结构的数据，进行传播
weight是传播概率
"""
import random
from copy import deepcopy


def IC(S, G):
    """
    实现独立级联模型。
    节点尝试激活它的所有邻居（每个只尝试激活一次），然后新激活的节点再尝试激活它们的邻居，
    重复该过程直到没有节点再可以被激活。
    """
    count = len(S)
    activity_set = set(S)
    active_nodes = set(S)
    while activity_set:
        new_activity_set = set()
        for seed in activity_set:
            neighbors = G.adj[seed]
            for node in neighbors.keys():
                weight = neighbors[node]['weight']
                if node not in active_nodes:
                    if random.random() < weight:
                        active_nodes.add(node)
                        new_activity_set.add(node)
        count = count + len(new_activity_set)
        activity_set = new_activity_set
    # print('count',count)
    return count


def LT(S, G):
    """
    实现线性阈值模型：
    节点周围的权值之和大于该节点的阈值，则该节点被激活。
    """
    count = len(S)
    activity_set = set(S)
    active_nodes = set(S)
    node_threshold = {}
    node_weights = {}
    while activity_set:
        new_activity_set = set()
        for seed in activity_set:
            neighbors = G.adj[seed]
            for node in neighbors.keys():
                weight = neighbors[node]['weight']
                if node not in active_nodes:
                    if node not in node_threshold:
                        node_threshold[node] = random.random()
                        node_weights[node] = 0
                    node_weights[node] += weight
                    if node_weights[node] >= node_threshold[node]:
                        active_nodes.add(node)
                        new_activity_set.add(node)
        count = count + len(new_activity_set)
        activity_set = new_activity_set
    return count


def IIC(S, G):
    """
    提升的独立级联模型，v被激活的概率使用
    1-(1-p)^l   l是邻居中被激活节点的个数
    :param S:
    :param G:
    :return:
    """
    T = deepcopy(S)  # 复制已经存在的节点
    i = 0
    while i < len(T):
        for v in G[T[i]]:  # 已选择节点的邻居节点
            if v not in T:  # 邻居还没有被选为种子节点
                v_adj = list(G[v].keys())  # v的邻居
                l = len([a for a in v_adj if a in T])  # v的邻居已经被激活的个数
                p = G[T[i]][v]['weight']  # 传播概率
                if random.random() <= 1 - (1 - p) ** l:  # if at least one of edges propagate influence
                    # print T[i], 'influences', v
                    T.append(v)
        i += 1
    return len(T)


def spread_run_IC(S, G, iterations):
    """
    使用network（自定义）网络结构的数据，求传播的平均影响范围
    :param S:
    :param G:
    :param Ep:
    :return:
    """
    avg = 0
    for i in range(iterations):
        avg = avg + float(IC(S, G)) / iterations
    return avg


def spread_run_IIC(S, G, iterations):
    """
    使用network（自定义）网络结构的数据，求传播的平均影响范围，使用IIC模型
    :param S:
    :param G:
    :param Ep:
    :return:
    """
    avg = 0
    for i in range(iterations):
        avg = avg + float(IIC(S, G)) / iterations
    return avg
