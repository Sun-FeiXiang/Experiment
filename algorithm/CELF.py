from igraph import *
from timeit import default_timer as timer
from algorithm.Spread.Networkx_spread import spread_run_IC
import networkx as nx


def CELF(G, k, p=0.1, mc=1000):
    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """
    # --------------------
    # 用贪心算法寻找第一个节点
    # --------------------
    # 计算第一次迭代排序列表
    start_time = time.time()
    marg_gain = [spread_run_IC(G, [node], p, mc) for node in G.nodes()]
    #print(marg_gain)
    # 创建节点及其边际收益的排序列表
    Q = sorted(zip(G.nodes(), marg_gain), key=lambda x: x[1], reverse=True)
    #print(Q)
    # 选择第一个节点并从候选列表中删除
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q, LOOKUPS, timelapse = Q[1:], [nx.number_of_nodes(G)], [time.time() - start_time]
    # --------------------
    # 使用列表排序过程查找下一个k-1节点
    # --------------------
    for _ in range(k - 1):
        check, node_lookup = False, 0
        while not check:
            # 计算传播计算的次数
            node_lookup += 1
            # 重新计算顶部节点的传播
            current = Q[0][0]
            # 评价传播函数并且存储边际收益到list中
            Q[0] = (current, spread_run_IC(G, S + [current], p, mc) - spread)
            # 重新对list进行排序
            Q = sorted(Q, key=lambda x: x[1], reverse=True)
            # 检查顶点排序后是否保持不变
            check = (Q[0][0] == current)

        # Select the next node
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)
        LOOKUPS.append(node_lookup)
        timelapse.append(time.time() - start_time)
        # 从list中移除已经选择的节点
        Q = Q[1:]

    return (S, SPREAD, timelapse, LOOKUPS)


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
    from algorithm.data_handle.read_Graph_networkx import read_Graph
    G = read_Graph("../data/graphdata/hep.txt")
    read_time = time.time()
    print('读取网络时间：', read_time - start)
    celf_output = CELF(G, 50, p=0.01, mc=1000)
    # print("greedy output: " + str(greedy_output[0]))
    list_IC_hep = []
    for k in range(1, 51):
        S = celf_output[0][:k]
        cur_spread = celf_output[1][k - 1]
        cal_time = celf_output[2][k - 1]
        print('newGreedyIC算法运行时间：', cal_time)
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
    df_IC_hep.to_csv('../data/output/greedy/IC_CELF_hep_Graph.csv')
    print('文件输出完毕——结束')
