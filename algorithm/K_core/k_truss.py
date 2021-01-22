import random

import networkx as nx
import sys


def k_truss(G):
    """
    求所有节点的k_truss值，针对无向图！！
    时间复杂度为：O(m*k^2+nk)
    :param G:
    :return:
    """
    edge_truss = dict()
    for edge in G.edges:  # O(m*k^2)
        start = edge[0]
        end = edge[1]
        start_adj = list(G.adj[start].keys())
        end_adj = list(G.adj[end].keys())
        intersection = [i for i in start_adj if i in end_adj]  # O(k^2)
        inter_node_num = 0
        for inter in intersection:
            inter_node_num = inter_node_num + min(G[start][inter]['weight'],G[end][inter]['weight'])
        edge_truss[edge] = inter_node_num + 2  # 边的truss值是交集+2
    # print(edge_truss[(0, 1)])
    node_truss = dict()
    for node in G.nodes:  # O(nk)
        end_node_list = list(G.adj[node].keys())
        max_truss = 0
        for end_node in end_node_list:
            if (node, end_node) in edge_truss.keys() and max_truss < edge_truss[(node, end_node)]:
                max_truss = edge_truss[(node, end_node)]
        node_truss[node] = max_truss

    return node_truss


def K_truss_sorted(G):
    node_truss = k_truss(G)
    node_truss = sorted(node_truss.items(), key=lambda A: A[1], reverse=True)
    #转换格式
    result = dict()
    for node_truss_line in node_truss:
        if node_truss_line[1] not in result.keys():
            result[node_truss_line[1]] = [node_truss_line[0]]
        else:
            result[node_truss_line[1]].append(node_truss_line[0])
    return result


def get_R_set(G, node):
    """
    生成节点的R集
    :param G:
    :param node:
    :return:
    """
    activity_set = list()
    activity_set.append(node)
    activity_nodes = list()
    activity_nodes.append(node)
    while activity_set:
        new_activity_set = list()
        for seed in activity_set:
            neightbors = G.adj[seed]
            for node in neightbors.keys():
                weight = neightbors[node]['weight']
                # print(node,weight)
                if node not in activity_nodes:
                    if random.random() < weight:
                        activity_nodes.append(node)
                        new_activity_set.append(node)
        activity_set = new_activity_set
    return activity_nodes


def get_local_influence(G, node_list):
    """
    获取局部影响力
    时间复杂度：O(k+mn)
    :param G:
    :param node:
    :return:按照节点影响力
    """
    node_inf_dict = dict()
    for node in node_list:
        stand = G.out_degree(node) - G.in_degree(node)  # 坚定系数，影响还是被影响 O(k)
        cur_R_set = get_R_set(G, node)  # O(mn)
        node_inf_dict[node] = stand * cur_R_set
    node_inf_list = sorted(node_inf_dict.items(), key=lambda A: A[1], reverse=True)  # 按照影响力排序
    result = [node[0] for node in node_inf_list]
    return result


if __name__ == "__main__":
    import time

    start = time.time()
    G = nx.read_weighted_edgelist("../../data/karate.txt", comments='#', nodetype=int, create_using=nx.DiGraph())
    read_time = time.time()
    print('读取网络时间：', read_time - start)
    # 生成固定的传播概率为0.01
    from generation.generation_propagation_probability import weight_probability_fixed

    weight_probability_fixed(G, 0.01)

    k_truss = k_truss(G)
    for a, b in k_truss.items():
        c = get_local_influence(G, b)
        print(c)
