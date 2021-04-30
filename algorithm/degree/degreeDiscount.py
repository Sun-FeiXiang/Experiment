"""
独立级联模型下，度折扣算法 degree discount heuristic [1]
[1] -- Wei Chen et al. Efficient influence maximization in Social Networks (algorithm 4)
"""
from model.ICM_nx import spread_run_IC,IC
from algorithm.priorityQueue import PriorityQueue as PQ  # priority queue
from timeit import default_timer as timer
from preprocessing.generation_propagation_probability import p_fixed,p_random
import networkx as nx


def degreeDiscountIC(G, k, p=.01):
    """
    在独立级联模型中查找要传播的初始节点集（带优先级队列）
    Input: G -- networkx 图对象
    k -- 需要的节点个数
    p -- 传播概率
    Output:
    S -- 选择的k个点的集合
    """
    S, timelapse, start_time = [], [], timer()
    dd = PQ()  # 度折扣
    t = dict()  #
    d = dict()  # 每个顶点的度

    # 初始度折扣
    for u in G.nodes():
        #d[u] = sum([G[u][v]['weight'] for v in G[u]])
        d[u] = len(G[u]) # each neighbor adds degree 1
        dd.add_task(u, -d[u])  # 添加每个节点的度数
        t[u] = 0

    # 贪心的给S加点
    for i in range(k):
        u, priority = dd.pop_item()  # 基于最大度折扣的节点提取 u及优先级代表节点及其度数
        S.append(u)
        timelapse.append(timer() - start_time)
        for v in G[u]:  # G[u]是u的邻接表
            if v not in S:  # ！！！
                #t[v] += G[u][v]['weight']
                t[v] += 1
                priority = d[v] - 2 * t[v] - (d[v] - t[v]) * t[v] * p
                dd.add_task(v, -priority)
    return (S, timelapse)


def degreeDiscountIC2(G, k, p=.01):
    """
    在独立级联模型中查找要传播的初始节点集（无优先级队列）
    Input: G -- networkx 图对象
    k -- 需要的节点个数
    p -- 传播概率
    Output:
    S -- 选择的k个点的集合
    Note: 该程序运行速度比使用PQ慢两倍。实施以验证结果
    """
    d = dict()
    dd = dict()  # 度折扣
    t = dict()  # 选择邻居的个数
    S,timelapse, start_time = [], [], timer()
    for u in G:
        d[u] = sum([G[u][v]['weight'] for v in G[u]])  # 每条边增加的度 一般是1，也有两个点之间有多条边
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd[u] = d[u]
        t[u] = 0
    for i in range(k):
        u, ddv = max(dd.items())
        dd.pop(u)
        S.append(u)
        timelapse.append(timer() - start_time)
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight']
                dd[v] = d[v] - 2 * t[v] - (d[v] - t[v]) * t[v] * p
    return (S, timelapse)


def degreeDiscountStar(G, k, p=.01):
    S, timelapse, start_time = [], [], timer()
    scores = PQ()
    d = dict()
    t = dict()
    for u in G:
        d[u] = sum([G[u][v]['weight'] for v in G[u]])
        t[u] = 0
        score = -((1 - p) ** t[u]) * (1 + (d[u] - t[u]) * p)
        scores.add_task(u, score)
    for iteration in range(k):
        u, priority = scores.pop_item()
        # print(iteration, -priority)
        S.append(u)
        timelapse.append(timer() - start_time)
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight']
                score = -((1 - p) ** t[u]) * (1 + (d[u] - t[u]) * p)
                scores.add_task(v, score)
    return (S, timelapse)


if __name__ == "__main__":
    import time
    start = time.time()
    from preprocessing.read_txt_nx import read_Graph
    # G = read_Graph('../../data/graphdata/hep.txt')
    G = nx.read_edgelist("../../data/graphdata/facebook_combined.txt", nodetype=int)  # 其他数据集使用此方式读取
    read_time = time.time()
    print('读取网络时间：', read_time - start)
    p = 0.03
    p_fixed(G,p)
    output = degreeDiscountIC(G, 50)
    list_IC_hep = []
    for k in range(1, 51):
        S = output[0][:k]
        cur_spread = IC(G, S,10000)
        cal_time = output[1][k - 1]
        print('degreeDiscount算法运行时间：', cal_time)
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
    df_IC_hep.to_csv('../../data/output/degreeDiscount/IC_degreeDiscount(p=0.03)_facebook.csv')
    print('文件输出完毕——结束')
