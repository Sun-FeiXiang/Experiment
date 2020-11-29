import networkx as nx
from timeit import default_timer as timer
import random
import multiprocessing as mp  # 多进程
import time
import getopt
import sys
import math
import heapq

"""

来源：Influence Maximization in Near-Linear Time: A Martingale Approach
算法步骤：
    第一步根据触发模型估算需要的反向可达集的数量并生成这些反向可达集（Sampling子函数），将他们存在一个数据结构R中；
    第二步是在R中用贪心方法找到k个节点使他们覆盖的反向可达集尽量多（NodeSelection子函数）。

"""


def create_worker(num):
    """
        创建进程
        :param num: 进程数目
        :param task_num: 分配给每个worker的任务数
    """
    global worker
    for i in range(num):
        worker.append(Worker(mp.Queue(), mp.Queue()))
        worker[i].start()


def finish_worker():
    """
    关闭所有子进程
    :return:
    """
    for w in worker:
        w.terminate()


class Worker(mp.Process):
    def __init__(self, inQ, outQ):
        super(Worker, self).__init__(target=self.start)
        self.inQ = inQ
        self.outQ = outQ
        self.R = []
        self.count = 0

    def run(self):

        while True:
            theta = self.inQ.get()
            # print(theta)
            while self.count < theta:
                print(node_num)
                v = random.randint(1, node_num)  # 生成多少个随机节点
                rr = generate_rr(v)
                self.R.append(rr)
                self.count += 1
            self.count = 0
            self.outQ.put(self.R)
            self.R = []


def IMM(G, k):
    return 1


def Sampling(G, D, k, epsilon, l):
    """
    第一步：估算需要的反向可达集的数量并生成这些反向可达集。
    :param G: networkx图
    :param D: 触发模型，LT、IC、...
    :param k: 预算k
    :param epsilon: 近似比参数
    :param l: 错误概率参数
    :return: 反向可达集序列R
    """
    R = []
    LB = 1
    n = len(G.nodes)  # 节点数目
    epsilon_p = epsilon * math.sqrt(2)  # epsilon'
    worker_num = 2
    create_worker(worker_num)
    for i in range(1, int(math.log2(n - 1)) + 1):
        s = time.time()
        x = n / (math.pow(2, i))
        lambda_p = ((2 + 2 * epsilon_p / 3) * (logcnk(n, k) + l * math.log(n) + math.log(math.log2(n))) * n) / pow(
            epsilon_p, 2)
        theta = lambda_p / x
        for ii in range(worker_num):
            worker[ii].inQ.put((theta - len(R)) / worker_num)
            for w in worker:
                R_list = w.outQ.get()
                R += R_list
            # finish_worker()
            # worker = []
            end = time.time()
            print('time to find rr', end - s)
            start = time.time()
            Si, f = node_selection(R, k)
            print(f)
            end = time.time()
            print('node selection time', time.time() - start)
            # print(F(R, Si))
            # f = F(R,Si)
            if n * f >= (1 + epsilon_p) * x:
                LB = n * f / (1 + epsilon_p)
                break
        # finish_worker()
        alpha = math.sqrt(l * math.log(n) + math.log(2))
        beta = math.sqrt((1 - 1 / math.e) * (logcnk(n, k) + l * math.log(n) + math.log(2)))
        lambda_aster = 2 * n * pow(((1 - 1 / math.e) * alpha + beta), 2) * pow(epsilon_p, -2)
        theta = lambda_aster / LB
        length_r = len(R)
        diff = theta - length_r
        # print(diff)
        _start = time.time()
        if diff > 0:
            # print('j')
            for ii in range(worker_num):
                worker[ii].inQ.put(diff / worker_num)
            for w in worker:
                R_list = w.outQ.get()
                R += R_list
        '''

        while length_r <= theta:
            v = random.randint(1, n)
            rr = generate_rr(v)
            R.append(rr)
            length_r += 1
        '''
        _end = time.time()
        # print(_end - _start)
        finish_worker()
        return R

    return R


def generate_rr_ic(G,node):
    """
    基于独立级联模型生成RR集
    :param node: 生成node的RR集
    :return: RR集
    """
    activity_set = list()
    activity_set.append(node)
    activity_nodes = list()
    activity_nodes.append(node)
    while activity_set:
        new_activity_set = list()
        for seed in activity_set:
            for node, weight in G.get_neighbors(seed):
                if node not in activity_nodes:
                    if random.random() < weight:
                        activity_nodes.append(node)
                        new_activity_set.append(node)
        activity_set = new_activity_set
    return activity_nodes


def generate_rr_lt(G,node):
    """
    基于线性阈值模型生成RR集
    :param node: 生成node的RR集
    :return: RR集
    """
    # calculate reverse reachable set using LT model
    # activity_set = list()
    activity_nodes = list()
    # activity_set.append(node)
    activity_nodes.append(node)
    activity_set = node

    while activity_set != -1:
        new_activity_set = -1

        neighbors = graph.get_neighbors(activity_set)
        if len(neighbors) == 0:
            break
        candidate = random.sample(neighbors, 1)[0][0]
        # print(candidate)
        if candidate not in activity_nodes:
            activity_nodes.append(candidate)
            # new_activity_set.append(candidate)
            new_activity_set = candidate
        activity_set = new_activity_set
    return activity_nodes


def logcnk(n, k):
    res = 0
    for i in range(n - k + 1, n + 1):
        res += math.log(i)
    for i in range(1, k + 1):
        res -= math.log(i)
    return res


def NodeSelection(G, k):
    """
    第二步：在R中用贪心方法找到k个节点使他们覆盖的反向可达集尽量多。
    :param G:
    :param k:
    :return:
    """
    S = []

    return S


"""
    定义全局变量:node_num、edge_num、graph、seeds
"""
node_num = 0  # 0
edge_num = 0

if __name__ == "__main__":
    import time

    start = time.time()
    from algorithm.graph_data_handle import read_gpickle

    G = read_gpickle("../data/graphs/hep.gpickle")
    node_num = len(G.nodes)
    edge_num = len(G.edges)

    read_time = time.time()
    print('读取网络时间：', read_time - start)

    # 生成固定的传播概率
    # from algorithm.generation_propagation_probability import fixed_probability
    # Ep = fixed_probability(G, 0.01)

    I = 1000

    list_IC_random_hep = []
    temp_time = timer()
    S = IMM(G, 10)
    # for k in range(5, 31, 5):
    #     S = IMM(G, k)
    #     cal_time = timer() - temp_time
    #     print('PageRank算法运行时间：', cal_time)
    #     print('k = ', k, '选取节点集为：', S)
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
    #
    # # df_IC_random_hep = pd.DataFrame(list_IC_random_hep)
    # # df_IC_random_hep.to_csv('../../data/output/IC_CCA_hep.csv')
    # # print('文件输出完毕——结束')
