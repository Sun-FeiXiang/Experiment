# 读取txt文件为networkx网络图

import networkx as nx
import random

def read_Graph(file_name, directed=False):
    """
    默认读取无向图
    :param file_name:
    :param directed:
    :return:
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    with open(file_name) as f:
        for line in f:
            if line[0] != '#':
                u, v = map(int, line.split())
                try:
                    G[u][v]['weight'] += 1
                except:
                    G.add_edge(u, v, weight=1)
    return G


# 计算平均度
def avg_degree(G):
    s = 0
    node_num = 0
    all_degree = 0.0
    for u in G:
        d = sum([float(G[u][v]['weight']) for v in G[u]])
        # print(cur_degree)
        node_num = node_num + 1
        all_degree = all_degree + d
    return all_degree / node_num

# 计算平均度，权重均为1
def avg_degree2(G):

    total_degree = 0
    for node in G.nodes:
        total_degree = total_degree + G.degree(node)

    return total_degree/nx.number_of_nodes(G)

if __name__ == "__main__":
    # G = read_Graph('../data/graphdata/phy.txt')
    # print("节点数：",nx.number_of_nodes(G),"边数：",nx.number_of_edges(G),"平均度：",avg_degree(G))
    # from dataPreprocessing.generation_propagation_probability import p_random
    # p_random(G)
    #
    G1 = nx.read_edgelist("../data/graphdata/DBLP.txt",nodetype=int)
    print("节点数：", nx.number_of_nodes(G1), "边数：", nx.number_of_edges(G1),"平均度",avg_degree2(G1))
    for node in G1.nodes:
        G1.add_node(node,time=random.random())
    for node in G1.nodes:
        print(G1.nodes)
