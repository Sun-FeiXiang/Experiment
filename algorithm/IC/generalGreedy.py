""" Implements greedy heuristic for IC model [1]

[1] -- Wei Chen et al. Efficient Influence Maximization in Social Networks (Algorithm 1)
"""
from algorithm.priorityQueue import PriorityQueue as PQ
from algorithm.IC.IC import runIC


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
