from algorithm.IMM.invgraph import Graph
from algorithm.IMM.graph import pGraph
import random
import multiprocessing as mp
import time
import math
from timeit import default_timer as timer
from algorithm.Spread.NetworkSpread import spread_run
"""

来源：Influence Maximization in Near-Linear Time: A Martingale Approach
算法步骤：
    第一步根据触发模型估算需要的反向可达集的数量并生成这些反向可达集（Sampling子函数），将他们存在一个数据结构R中；
    第二步是在R中用贪心方法找到k个节点使他们覆盖的反向可达集尽量多（NodeSelection子函数）。
"""


def create_worker(num,task_num):
    """
        创建进程
        :param num: 进程数目
        :param task_num: 分配给每个worker的任务数
    """
    global worker
    for i in range(num):
        worker.append(Worker(mp.Queue(), mp.Queue(),task_num))
        worker[i].start()


def finish_worker():
    """
    关闭所有子进程
    :return:
    """
    for w in worker:
        w.terminate()


class Worker(mp.Process):
    def __init__(self, inQ, outQ,task_num):
        super(Worker, self).__init__(target=self.start)
        self.inQ = inQ
        self.outQ = outQ
        self.R = []
        self.count = 0
        self.node_num = task_num

    def run(self):

        while True:
            theta = self.inQ.get()
            # print(theta)
            while self.count < theta:
                #print('node_num',self.node_num)
                v = random.randint(1, self.node_num)  # 生成随机节点
                rr = generate_rr(v) #生成节点v的RR集
                self.R.append(rr)
                self.count += 1
            self.count = 0
            self.outQ.put(self.R)
            self.R = []


def sampling(epsoid, l):
    print("Sampling ...")
    global graph, seed_size, worker
    R = []
    LB = 1
    n = node_num
    #print('sampling',node_num)
    k = seed_size
    epsoid_p = epsoid * math.sqrt(2)
    worker_num = 2
    create_worker(worker_num,node_num)
    for i in range(1, int(math.log2(n - 1)) + 1):
        s = time.time()
        x = n / (math.pow(2, i))
        lambda_p = ((2 + 2 * epsoid_p / 3) * (logcnk(n, k) + l * math.log(n) + math.log(math.log2(n))) * n) / pow(
            epsoid_p, 2)
        theta = lambda_p / x
        # print(theta-len(R))
        for ii in range(worker_num):
            worker[ii].inQ.put((theta - len(R)) / worker_num)
        for w in worker:
            R_list = w.outQ.get()
            R += R_list
        # finish_worker()
        # worker = []
        end = time.time()
        #print('   RR集计算时间：', end - s)
        start = time.time()
        Si, f = node_selection(R, k)
        #print(f)
        end = time.time()
        #print('   节点选择时间：', time.time() - start)
        # print(F(R, Si))
        # f = F(R,Si)
        if n * f >= (1 + epsoid_p) * x:
            LB = n * f / (1 + epsoid_p)
            break
    # finish_worker()
    alpha = math.sqrt(l * math.log(n) + math.log(2))
    beta = math.sqrt((1 - 1 / math.e) * (logcnk(n, k) + l * math.log(n) + math.log(2)))
    lambda_aster = 2 * n * pow(((1 - 1 / math.e) * alpha + beta), 2) * pow(epsoid, -2)
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


def generate_rr(v):
    global model
    if model == 'IC':
        return generate_rr_ic(v)
    elif model == 'LT':
        return generate_rr_lt(v)


def node_selection(R, k):
    """
    第二步
    :param R: 反向可达集
    :param k: 初始节点数
    :return: 影响最大的节点集
    """
    Sk = set()
    rr_degree = [0 for ii in range(node_num + 1)]
    node_rr_set = dict()
    # node_rr_set_copy = dict()
    matched_count = 0
    for j in range(0, len(R)):
        rr = R[j]
        for rr_node in rr:
            # print(rr_node)
            rr_degree[rr_node] += 1
            if rr_node not in node_rr_set:
                node_rr_set[rr_node] = list()
                # node_rr_set_copy[rr_node] = list()
            node_rr_set[rr_node].append(j)
            # node_rr_set_copy[rr_node].append(j)
    for i in range(k):
        max_point = rr_degree.index(max(rr_degree))
        Sk.add(max_point)
        matched_count += len(node_rr_set[max_point])
        index_set = []
        for node_rr in node_rr_set[max_point]:
            index_set.append(node_rr)
        for jj in index_set:
            rr = R[jj]
            for rr_node in rr:
                rr_degree[rr_node] -= 1
                node_rr_set[rr_node].remove(jj)
    return Sk, matched_count / len(R)


'''
def node_selection(R, k):
    # use CELF to accelerate
    Sk = set()
    node_rr_set = dict()
    rr_degree = [0 for ii in range(node_num + 1)]
    matched_count = 0
    for i, rr in enumerate(R):
        for v in rr:
            if v in node_rr_set:
                node_rr_set[v].add(i)
                rr_degree[v] += 1
            else:
                node_rr_set[v] = {i}
    max_heap = list()
    for key, value in node_rr_set.items():
        max_heap.append([-len(value), key, 0])
    heapq.heapify(max_heap)
    i = 0
    covered_set = set()
    while i < k:
        val = heapq.heappop(max_heap)
        if val[2] != i:
            node_rr_set[val[1]] -= covered_set
            val[0] = -len(node_rr_set[val[1]])
            val[2] = i
            heapq.heappush(max_heap, val)
        else:
            Sk.add(val[1])
            i += 1
            covered_set |= node_rr_set[val[1]]
    return Sk, len(covered_set) / len(R)
'''


def generate_rr_ic(node):
    activity_set = list()
    activity_set.append(node)
    activity_nodes = list()
    activity_nodes.append(node)
    while activity_set:
        new_activity_set = list()
        for seed in activity_set:
            for node, weight in graph.get_neighbors(seed):
                if node not in activity_nodes:
                    if random.random() < weight:
                        activity_nodes.append(node)
                        new_activity_set.append(node)
        activity_set = new_activity_set
    return activity_nodes


def generate_rr_lt(node):
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


def imm(epsoid, l):
    n = node_num
    k = seed_size
    l = l * (1 + math.log(2) / math.log(n))
    R = sampling(epsoid, l)
    Sk, z = node_selection(R, k)
    return Sk


def logcnk(n, k):
    res = 0
    for i in range(n - k + 1, n + 1):
        res += math.log(i)
    for i in range(1, k + 1):
        res -= math.log(i)
    return res


def read_file(network):
    """
    读取网络数据并用自定义的数据结构存储
    :param network: 文件路径
    """
    global node_num, edge_num, graph
    data_lines = open(network, 'r').readlines()
    node_num = int(data_lines[0].split()[0])
    edge_num = int(data_lines[0].split()[1])

    for data_line in data_lines[1:]:
        start, end, weight = data_line.split()
        graph.add_edge(int(start), int(end), float(weight))

"""
    定义全局变量:node_num、edge_num、graph、seeds
"""
node_num = 0  # 0
edge_num = 0
graph = Graph()
pGraph = pGraph()
model = 'IC'

if __name__ == "__main__":
    start = time.time()
    network_path = "test_data/NetHEPT.txt"
    model = 'IC'
    seed_size = 50
    termination = 10
    read_file(network_path)
    read_time = time.time()
    print('读取网络时间：', read_time - start)
    temp_time = timer()
    print("(节点数，边数)",node_num, edge_num)

    worker = []
    epsoid = 0.5
    l = 1
    I = 1000    #迭代次数
    S = imm(epsoid, l)
    cal_time = timer() - temp_time
    print('IMM算法运行时间：', cal_time)
    print('k = ', seed_size, '选取节点集为：', S)

    list_IC_random_hep = []
    iterations = 1000
    average_cover_size = spread_run(S,graph,iterations)
    list_IC_random_hep.append({'k': seed_size,'run time': cal_time,'average cover size': average_cover_size,'S': S})
    temp_time = timer()  # 记录当前时间
    print('平均覆盖大小：', average_cover_size)
    # import pandas as pd
    #
    # # df_IC_random_hep = pd.DataFrame(list_IC_random_hep)
    # # df_IC_random_hep.to_csv('../../data/output/IC_CCA_hep.csv')
    # # print('文件输出完毕——结束')
