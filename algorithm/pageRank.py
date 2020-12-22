from timeit import default_timer as timer
import networkx as nx

"""
PageRank作为一个经典的网页排序算法，在影响最大化中的应用，一般作为一个对照实验。本处的实现主要依据下面论文中的设置
来源：Scalable Influence Maximization for Prevalent Viral Marketing in Large-Scale Social Networks∗
对比实验
"""


def pageRank(G, k, p=.01):
    pages_rank = nx.pagerank(G, tol=1e-4)  # 输入是有向图
    pages_rank = sorted(pages_rank.items(), key=lambda x: x[1], reverse=True)
    S = []
    for u, ranking in pages_rank:
        if len(S) == k:
            break
        S.append(u)
    return S


if __name__ == "__main__":
    import time

    start = time.time()
    G = nx.read_weighted_edgelist("../data/soc-Epinions1.txt", comments='#', nodetype=int, create_using=nx.DiGraph())
    G1 = G.copy()
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    #生成固定的传播概率
    from generation.generation_propagation_probability import weight_probability_inEdge
    weight_probability_inEdge(G)

    I = 1000

    list_IC_random_hep = []
    temp_time = timer()
    for k in range(5, 51, 5):
        S = pageRank(G, k)
        cal_time = timer() - temp_time
        print('PageRank算法运行时间：', cal_time)
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
    df_IC_random_hep.to_csv('../data/output/pageRank/IC_pageRank_Epinions.csv')
    print('文件输出完毕——结束')
