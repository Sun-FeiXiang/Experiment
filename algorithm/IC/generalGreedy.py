"""
算法：IC模型中的greedy heuristic（每个贪心算法的具体贪心策略不同）
来源：Wei Chen et al. Efficient Influence Maximization in Social Networks (算法 1)
"""
from algorithm.priorityQueue import PriorityQueue as PQ
from algorithm.IC.IC import runIC
import networkx as nx
from timeit import default_timer as timer

def generalGreedy(G, k, p=.01):
    """ 使用一般的贪心启发式寻找初始节点集S
    输入: G -- networkx图对象
    k -- 初始需要的节点数量
    p -- 传播概率
    输出: S -- 要传播的k个节点的初始集合
    """
    import time
    start = time.time()
    R = 20  # 随机级联运行的次数
    S = []  # 选择的节点集
    # 如果当前选定的节点达到最大传播，则将节点添加到S
    for i in range(k):
        s = PQ()  # 优先队列
        for v in G.nodes():
            if v not in S:
                s.add_task(v, 0)  # 初始传播值
                for j in range(R):  # 运行R次随机级联
                    [priority, count, task] = s.entry_finder[v]
                    s.add_task(v, priority - float(len(runIC(G, S + [v], p))) / R)  # 加入标准传播值
        task, priority = s.pop_item()
        S.append(task)
        # print(i, k, time.time() - start)
    return S



if __name__ == "__main__":
    import time

    start = time.time()
    G = nx.read_weighted_edgelist("../../data/NetHEPT.txt", comments='#', nodetype=int, create_using=nx.DiGraph())
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    # 生成固定的传播概率为0.01
    from generation.generation_propagation_probability import weight_probability_fixed

    weight_probability_fixed(G, 0.01)

    I = 1000

    list_IC_random_hep = []
    temp_time = timer()
    for k in range(1, 51):
        S = generalGreedy(G, k)
        cal_time = timer() - temp_time
        print('generalGreedy算法运行时间：', cal_time)
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
    df_IC_random_hep.to_csv('../../data/output/greedy/IC_generalGreedy_NetHEPT.csv')
    print('文件输出完毕——结束')
