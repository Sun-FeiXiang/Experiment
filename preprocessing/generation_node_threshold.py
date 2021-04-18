"""
生成节点阈值
"""
import networkx as nx
import random


def random_threshold(G):
    """
    生成随机阈值
    :param G:
    :return:
    """
    node_threshold = dict()
    for node in G.nodes:
        node_threshold[node] = random.random()
    return node_threshold
