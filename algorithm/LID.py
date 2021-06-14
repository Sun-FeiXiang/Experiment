"""
Identification of influencers in complex networks by local information dimensionality
LID

"""

import networkx as nx
from preprocessing.read_txt_nx import read_Graph
from preprocessing.generation_propagation_probability import fixed_weight,fixed_distance

def get_central_node(G):
    return nx.floyd_warshall(G,weight="distance")


def LID():
    return 1


if __name__ == "__main__":
    G = read_Graph("../data/graphdata/hep.txt", directed=False)
    # G = nx.read_edgelist("../../data/graphdata/email.txt", nodetype=int, create_using=nx.Graph)  # 其他数据集使用此方式读取
    # fixed_weight(G)
    fixed_distance(G)
    a = get_central_node(G)
    for key, value in a.items():
        print('%s %s' % (key, value.values()))