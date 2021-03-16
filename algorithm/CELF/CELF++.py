from diffusion.Networkx_diffusion import spread_run_IC
from heapdict import heapdict
from igraph import *
from timeit import default_timer as timer


class Node(object):
    def __init__(self, node):
        self.node = node
        self.mg1 = 0
        self.prev_best = None
        self.mg2 = 0
        self.flag = None
        self.list_index = 0


def CELFpp(G, k, p=0.1, mc=1000):
    S = list()
    spread, timelapse, start_time = [], [], time.time()
    Q = heapdict()
    last_seed = None
    cur_best = None
    node_data_list = []

    for node in G.nodes:
        node_data = Node(node)
        node_data.mg1 = spread_run_IC(G, [node], p, mc)  # 该节点的边际收益
        node_data.prev_best = cur_best
        node_data.mg2 = spread_run_IC(G, [node, cur_best.node], p, mc) if cur_best else node_data.mg1
        node_data.flag = 0
        cur_best = cur_best if cur_best and cur_best.mg1 > node_data.mg1 else node_data
        G.nodes[node]['node_data'] = node_data
        node_data_list.append(node_data)
        node_data.list_index = len(node_data_list) - 1
        Q[node_data.list_index] = - node_data.mg1

    while len(S) < k:
        node_idx, _ = Q.peekitem()
        node_data = node_data_list[node_idx]
        if node_data.flag == len(S):
            S.append(node_data.node)
            if len(spread) == 0:
                spread.append(node_data.mg1)
            else:
                spread.append(spread[-1]+node_data.mg1)
            timelapse.append(time.time() - start_time)
            del Q[node_idx]
            last_seed = node_data
            continue
        elif node_data.prev_best == last_seed:
            node_data.mg1 = node_data.mg2
        else:
            before = spread_run_IC(G, S, p, mc)
            S.append(node_data.node)
            after = spread_run_IC(G, S, p, mc)
            S.remove(node_data.node)
            node_data.mg1 = after - before
            node_data.prev_best = cur_best
            S.append(cur_best.node)
            before = spread_run_IC(G, S, p, mc)
            S.append(node_data.node)
            after = spread_run_IC(G, S, p, mc)
            S.remove(cur_best.node)
            if node_data.node != cur_best.node:
                S.remove(node_data.node)
            node_data.mg2 = after - before

        if cur_best and cur_best.mg1 < node_data.mg1:
            cur_best = node_data

        node_data.flag = len(S)
        Q[node_idx] = - node_data.mg1
    return (S, spread, timelapse)


def test_igraph():
    # Create simple network with 0 and 1 as the influential nodes
    source = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5]
    target = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 6, 7, 8, 9]

    g = Graph(directed=True)
    g.add_vertices(range(10))
    g.add_edges(zip(source, target))

    # Plot graph
    g.vs["label"], g.es["color"], g.vs["color"] = range(10), "#B3CDE3", "#FBB4AE"
    plot(g, bbox=(200, 200), margin=20, layout=g.layout("kk"))
    return g


if __name__ == "__main__":
    import time

    start = time.time()
    from dataPreprocessing.read_txt_nx import read_Graph

    G = read_Graph("../../data/graphdata/hep.txt")
    read_time = time.time()
    print('读取网络时间：', read_time - start)
    CELFpp_output = CELFpp(G, 50, p=0.01, mc=1000)
    # print("greedy output: " + str(greedy_output[0]))
    list_IC_hep = []
    for k in range(1, 51):
        S = CELFpp_output[0][:k]
        cur_spread = CELFpp_output[1][k - 1]
        cal_time = CELFpp_output[2][k - 1]
        print('CELF++算法运行时间：', cal_time)
        print('k = ', k, '选取节点集为：', S)
        print('k=', k, '平均覆盖大小：', cur_spread)
        list_IC_hep.append({
            'k': k,
            'run time': cal_time,
            'average cover size': cur_spread,
            'S': S
        })
        temp_time = timer()  # 记录当前时间
    import pandas as pd
    df_IC_hep = pd.DataFrame(list_IC_hep)
    df_IC_hep.to_csv('../../data/output/greedy/IC_CELF++_hep_Graph.csv')
    print('文件输出完毕——结束')
