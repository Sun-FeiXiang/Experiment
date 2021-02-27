"""
基于支配集 测试不同算法的文件
"""

from timeit import default_timer as timer
from networkx.algorithms.dominating import dominating_set


def get_dominating_set_subGraph(G):
    min_dominating_set = dominating_set(G)
    subGraph = G.copy()
    for node in G.nodes:
        if node not in min_dominating_set:
            subGraph.remove_node(node)
    return subGraph


if __name__ == '__main__':
    import time

    start = time.time()
    from dataPreprocessing.read_gpickle_nx import read_gpickle_DiGraph

    G = read_gpickle_DiGraph("../../data/graphs/hep.gpickle")
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    # 生成固定的传播概率
    from dataPreprocessing.generation_propagation_probability import fixed_probability
    Ep = fixed_probability(G, 0.01)

    time_begin_dominating_set = time.time()
    sub_g = get_dominating_set_subGraph(G)
    print('求最小支配集构成的子图时间：', time.time()- time_begin_dominating_set)

    I = 1000

    from algorithm.IC.newGreedyIC import newGreedyIC

    list_IC_random_hep = []
    temp_time = timer()
    for k in range(5, 31, 5):
        S = newGreedyIC(sub_g, k,Ep)
        cal_time = timer() - temp_time
        print('算法运行时间：', cal_time)
        print('选取节点集为：', S)

        from algorithm.IC.IC import avgIC_cover_size

        average_cover_size = avgIC_cover_size(G, S, 0.01, I)
        print('平均覆盖大小：', average_cover_size)

        list_IC_random_hep.append({
            'k': k,
            'run time': cal_time,
            'average cover size': average_cover_size,
            'S': S
        })
        temp_time = timer()  # 记录当前时间

    import pandas as pd
    df_IC_random_hep = pd.DataFrame(list_IC_random_hep)
    df_IC_random_hep.to_csv('../../data/output/IC_newGreedyIC_hep_dominating_set.csv')
    print('文件输出完毕——结束')
