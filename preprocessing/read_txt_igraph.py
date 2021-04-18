from igraph import *
import time


def read_Graph(file_name, directed=False):
    """
    默认读取无向图
    :param file_name:
    :param directed:
    :return:
    """
    # 注意igraph不能在没有节点的情况下增加边

    if directed:
        g = Graph(directed=True)
    else:
        g = Graph(directed=False)

    file_content = dict()
    with open(file_name) as f:
        # 添加节点
        for line in f:
            if line[0] != '#':
                u, v = map(int, line.split())
                try:
                    file_content[(u, v)] += 1
                except:
                    file_content[(u, v)] = 1
            else:
                # 只处理第一行为 节点数 边数 的数据
                vertex, edge = map(int, line[1:].split())
                g.add_vertices(vertex)
    g.add_edges(file_content.keys())
    # 权重只能另外加入
    for e, w in zip(g.es, file_content.values()):
        e['weight'] = w
    return g


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


if __name__ == "__main__":
    start_time = time.time()
    G = read_Graph('data/hep.txt')
    end_time = time.time()
    print((end_time-start_time))
    print(G.neighbors(2))
    # G = Graph.Read_Edgelist('data/hep.txt',directed=False)
    # print(timer()-end_time)
    # print(G.degree(131))
    # for v in G.:
    #     print(v)

    # 获取某条边的权重
    # for index,e in enumerate(G.es):
    #     print(index,e)