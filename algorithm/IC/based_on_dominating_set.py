"""
基于支配集 测试不同算法的文件
"""
import networkx as nx
from timeit import default_timer as timer
from networkx.algorithms import approximation as apxa

def get_dominating_set_subGraph(G):
    min_dominating_set = apxa.min_edge_dominating_set(G)
    subGraph = G
    for node in G.nodes:
        if node not in min_dominating_set:
            subGraph.remove_node(node)
    return subGraph


if __name__ == '__main__':
    import time

    start = time.time()
    from algorithm.graph_data_handle import read_gpickle

    G = read_gpickle("../../data/graphs/hep.gpickle")
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    # 生成固定的传播概率
    # from algorithm.generation_propagation_probability import fixed_probability
    # Ep = fixed_probability(G, 0.01)

    print('求最小支配集：')

    print(min_dominating_set)

    # I = 1000
    #
    # from algorithm.IC.randomHeuristic import randomHeuristic
    #
    # list_IC_random_hep = []
    # temp_time = timer()
    # for k in range(5, 31, 5):
    #     S = randomHeuristic(sub_g, k)
    #     cal_time = timer() - temp_time
    #     print('算法运行时间：', cal_time)
    #     print('选取节点集为：', S)
    #
    #     from algorithm.IC.IC import avgIC_cover_size
    #
    #     average_cover_size = avgIC_cover_size(G, S, 0.01, I)
    #     print('平均覆盖大小：', average_cover_size)
    #
    #     list_IC_random_hep.append({
    #         'k': k,
    #         'run time': cal_time,
    #         'average cover size': average_cover_size,
    #         'S': S
    #     })
    #     temp_time = timer()  # 记录当前时间
    #
    # import pandas as pd
    # df_IC_random_hep = pd.DataFrame(list_IC_random_hep)
    # df_IC_random_hep.to_csv('../../data/output/IC_random_hep.csv')
    # print('文件输出完毕——结束')
