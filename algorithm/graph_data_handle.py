import networkx as nx
import matplotlib.pyplot as plt


def read_gpickle(file):
    """
    :param file: gpickle文件路径
    :return: 处理为标准的networkx对象 有向图
    """
    g_gpickle = nx.read_gpickle(file)
    G = nx.DiGraph()
    for key, values in g_gpickle.edge.items():
        for end in list(values.keys()):
            # print(key, end, values[end]['weight'])
            G.add_edge(key, end, weight=values[end]['weight'])
    #nx.write_weighted_edgelist(G, "test.weighted.edgelist")
    return G


if __name__ == "__main__":
    G = read_gpickle("../data/graphs/hep.gpickle")
