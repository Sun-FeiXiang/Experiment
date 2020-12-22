"""
基于独立级联模型random heuristic[1]
随机均匀取k个节点

[1] -- Wei Chen et al. Efficient influence maximization in Social Networks
"""
import networkx as nx
from timeit import default_timer as timer


def randomHeuristic(G, k, p=.01):
    """
    在独立级联模型下找到初始传播的k个点
    输入: G -- networkx图对象
    k -- 需要的节点数
    p -- 传播概率
    输出:
    S -- 选择的k个点的集合
    """
    import random
    S = random.sample(G.nodes(), k)
    return S


if __name__ == "__main__":
    import time

    start = time.time()
    G = nx.read_weighted_edgelist("../../data/NetPHY.txt", comments='#', nodetype=int, create_using=nx.Graph())
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    # 生成固定的传播概率为0.01
    from generation.generation_propagation_probability import weight_probability_fixed
    weight_probability_fixed(G, 0.01)

    I = 1000

    list_IC_random_hep = []
    temp_time = timer()
    for k in range(1, 51):
        S = randomHeuristic(G, k)
        cal_time = timer() - temp_time
        print('randomHeuristic算法运行时间：', cal_time)
        print('k = ', k, '选取节点集为：', S)

        from algorithm.Spread.NetworkxSpread import spread_run_IC

        average_cover_size = spread_run_IC(S, G, 1000)
        print('k=', k, '平均覆盖大小：', average_cover_size)

        list_IC_random_hep.append({
            'k': k,
            'run time': cal_time,
            'average cover size': average_cover_size,
            'S': S
        })
        temp_time = timer()  # 记录当前时间

    import pandas as pd

    df_IC_random_hep = pd.DataFrame(list_IC_random_hep)
    df_IC_random_hep.to_csv('../../data/output/random/IC_random_NetPHY_Graph.csv')
    print('文件输出完毕——结束')
