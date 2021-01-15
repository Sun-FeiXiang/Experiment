import networkx as nx


def node_core_number(G):
    G.remove_edges_from(nx.selfloop_edges(G))
    node_core_number = nx.core_number(G)  # 每个顶点的core值

    return node_core_number


if __name__ == "__main__":
    import time

    start = time.time()
    G = nx.read_weighted_edgelist("../../data/NetHEPT.txt", comments='#', nodetype=int, create_using=nx.DiGraph())

    read_time = time.time()
    print('读取网络时间：', read_time - start)

    # 生成固定的传播概率为0.01
    from generation.generation_propagation_probability import weight_probability_fixed

    weight_probability_fixed(G, 0.01)

    node_core_number = node_core_number(G)