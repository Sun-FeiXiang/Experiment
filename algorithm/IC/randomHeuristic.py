"""
基于独立级联模型random heuristic[1]
随机均匀取k个节点

[1] -- Wei Chen et al. Efficient influence maximization in Social Networks
"""
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
    from dataPreprocessing.read_txt_nx import read_Graph
    G = read_Graph('../../data/graphdata/phy.txt')
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    I = 1000

    list_IC_hep = []
    temp_time = timer()
    k = 50
    S = randomHeuristic(G, k)
    cal_time = timer() - temp_time

    for i in range(1, 51):
        s = S[0:i]
        run_time = cal_time/50*(i+1)
        print('randomHeuristic算法运行时间：', run_time)
        print('k = ', i, '选取节点集为：', s)

        from diffusion import spread_run_IC

        average_cover_size = spread_run_IC(G,s,0.01, 1000)
        print('k=', i, '平均覆盖大小：', average_cover_size)

        list_IC_hep.append({
            'k': i,
            'run time': run_time,
            'average cover size': average_cover_size,
            'S': s
        })
        temp_time = timer()  # 记录当前时间

    import pandas as pd

    df_IC_random_hep = pd.DataFrame(list_IC_hep)
    df_IC_random_hep.to_csv('../../data/output/random/IC_random_phy_Graph.csv')
    print('文件输出完毕——结束')
