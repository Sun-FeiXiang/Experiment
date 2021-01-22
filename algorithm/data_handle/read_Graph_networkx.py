import networkx as nx


def read_Graph(file_name,directed=False):
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


def avg_degree(G):
    s = 0
    node_num = 0
    all_degree = 0.0
    for u in G:
        d = sum([float(G[u][v]['weight']) for v in G[u]])
        # print(cur_degree)
        node_num = node_num + 1
        all_degree = all_degree + d
    return all_degree/node_num


if __name__ == "__main__":
    G = read_Graph('../../data/graphdata/hep.txt')
    a = avg_degree(G)