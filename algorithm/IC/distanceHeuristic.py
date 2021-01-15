"""
算法：基于独立级联模型的degree heuristic
    获取拥有最大度的前k个节点
来源：Wei Chen et al. Efficient influence maximization in Social Networks
"""
import networkx as nx
from algorithm.priorityQueue import PriorityQueue as PQ  # 优先队列
from timeit import default_timer as timer


def distanceHeuristic(G, k, closeness_centrality, p=.01):
    """
    在独立级联模型中查找要传播的初始节点集（带优先级队列）
    输入: G -- networkx图对象
    k -- 需要的节点数
    p -- 传播概率
    输出:
    S -- 选择的k个点的集合
    """
    S = []
    d = PQ()
    for key in closeness_centrality.keys():
        d.add_task(key, -closeness_centrality[key])
    for i in range(k):
        u, priority = d.pop_item()
        S.append(u)
    return S


if __name__ == "__main__":
    import time

    start = time.time()
    G = nx.read_weighted_edgelist("../../data/graphdata/hep.txt", comments='#', nodetype=int, create_using=nx.Graph())
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    # 生成固定的传播概率为0.01
    from generation.generation_propagation_probability import weight_probability_fixed

    weight_probability_fixed(G, 0.01)

    closeness_centrality = nx.closeness_centrality(G)

    I = 1000

    list_IC_random_hep = []
    temp_time = timer()
    # S = distanceHeuristic(G, 10)
    for k in range(1, 51):
        S = distanceHeuristic(G, k, closeness_centrality)
        cal_time = timer() - temp_time
        print('distanceHeuristic算法运行时间：', cal_time)
        print('k = ', k, '选取节点集为：', S)

        from algorithm.Spread.Networkx_spread import spread_run_IIC

        average_cover_size = spread_run_IIC(S, G, 1000)
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
    df_IC_random_hep.to_csv('../../data/output/distance/IIC_distance_hep_Graph.csv')
    print('文件输出完毕——结束')
