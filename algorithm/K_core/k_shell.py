import networkx as nx


def node_core_number(G):
    """
   获得图中所有节点的core值
   :param G:
   :return:
   """
    E = G.copy()

    k_s = {}
    k = 0

    while nx.number_of_nodes(E) > 0:
        nodes = get_node_degree(E, k)  # E中度为k的节点集
        E.remove_nodes_from(nodes)  # 从图中移除该节点集
        if len(get_node_degree(E, k)) == 0:
            k = k + 1


def get_node_degree(G, d):
    """
    获得所有节点的度为d的节点
    时间复杂度：O(n)
    :param G:
    :param d:
    :return:
    """
    nodes = []
    for u in G.nodes():
        cur_d = sum([G[u][v]['weight'] for v in G[u]])
        if cur_d == d:
            nodes.append(u)
    return nodes


if __name__ == "__main__":
    import time

    start = time.time()
    from dataPreprocessing.read_txt_nx import read_Graph

    G = read_Graph("../../data/NetHEPT.txt")
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    node_core_number = node_core_number(G)
