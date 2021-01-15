import numpy as np


def IC(g, S, p=0.5, mc=1000):
    """
    Input:  graph object, set of seed nodes, propagation probability
            and the number of Monte-Carlo simulations
    Output: average number of nodes influenced by the seed nodes
    """

    # Loop over the Monte-Carlo Simulations
    spread = []
    for i in range(mc):
        # Simulate propagation process
        new_active, A = S[:], S[:]
        while new_active:
            new_ones = []
            for node in new_active:
                np.random.seed(i)  # 随机数种子
                success = np.random.uniform(0, 1, len(
                    g.neighbors(node, mode="out"))) < p  # 生成（出邻居个数的随机数），并判断他们是否小于传播概率，返回的是[Tru,False,...]
                new_ones += list(np.extract(success, g.neighbors(node, mode="out")))  # 提取success中为True的节点
            new_active = list(set(new_ones) - set(A))
            # 添加新激活的节点到激活节点集中
            A += new_active
        spread.append(len(A))

    return np.mean(spread)
