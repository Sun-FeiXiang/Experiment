"""
数据处理
"""
import networkx as nx
import math


def get_all_around_value(G):
    all_around_value = dict()
    protein_cores = nx.core_number(G)  # 每个顶点的core值
    betweenness_centrality = nx.betweenness_centrality(G)
    for node in G.nodes:
        k = G.degree(node)
        k_s = protein_cores[node]
        c_b = betweenness_centrality[node]
        all_around_value[node] = math.sqrt(math.pow(k,2)+math.pow(k_s,2)+math.pow(c_b,2))
    return all_around_value

