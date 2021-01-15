import random

"""
使用自定义的network数据结构进行传播。
"""


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
            for node, weight in G.get_neighbors(seed):
                if node not in active_nodes:
                    if random.random() < weight:
                        active_nodes.add(node)
                        new_activity_set.add(node)
        count += len(new_activity_set)
        activity_set = new_activity_set
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
            for node, weight in G.get_neighbors(seed):
                if node not in active_nodes:
                    if node not in node_threshold:
                        node_threshold[node] = random.random()
                        node_weights[node] = 0
                    node_weights[node] += weight
                    if node_weights[node] >= node_threshold[node]:
                        active_nodes.add(node)
                        new_activity_set.add(node)
        count += len(new_activity_set)
        activity_set = new_activity_set
    return count


def spread_run(S, G, iterations):
    """
    使用network（自定义）网络结构的数据，求传播的平均影响范围
    :param S:
    :param G:
    :param Ep:
    :return:
    """
    avg = 0
    for i in range(iterations):
        avg += float(IC(S, G)) / iterations
    return avg
