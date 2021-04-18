# 读取gpickle文件为networkx网络图
import networkx as nx


def read_gpickle_DiGraph(file):
    """
    :param file: gpickle文件路径
    :return: 处理为标准的networkx对象 有向图
    """
    g_gpickle = nx.read_gpickle(file)
    G = nx.DiGraph()
    for key, values in g_gpickle.edge.items():
        for end in list(values.keys()):
            G.add_edge(key, end, weight=values[end]['weight'])
    return G


def read_gpickle_Graph(file):
    """
    :param file: gpickle文件路径
    :return: 处理为标准的networkx对象 有向图
    """
    g_gpickle = nx.read_gpickle(file)
    G = nx.Graph()
    for key, values in g_gpickle.edge.items():
        for end in list(values.keys()):
            G.add_edge(key, end, weight=values[end]['weight'])
    return G


if __name__ == "__main__":
    G = read_gpickle_DiGraph("../data/graphs/hep.gpickle")
