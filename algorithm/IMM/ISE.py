from algorithm.IMM.invgraph import Graph
from algorithm.IMM.graph import pGraph
import random
import multiprocessing as mp
import time


class Worker(mp.Process):
    def __init__(self, outQ, count):
        super(Worker, self).__init__(target=self.start)
        self.outQ = outQ
        self.count = count
        self.sum = 0

    def run(self):
        while self.count > 0:
            # print(self.count)
            res = ise()
            self.sum += res
            self.count -= 1
            if self.count == 0:
                self.outQ.put(self.sum)


def create_worker(num, task_num):
    """
        创建进程
        :param num: 进程数
        :param task_num: 分配给每个worker的任务数
    """
    for i in range(num):
        worker.append(Worker(mp.Queue(), task_num))
        worker[i].start()


def finish_worker():
    """
    关闭所有子进程
    :return:
    """
    for w in worker:
        w.terminate()


def ise():
    print('model', model)
    print('Seeds', seeds)
    if model == 'IC':
        return IC()
    elif model == 'LT':
        return LT()


def read_file(network, seed):
    """
       读取网络数据并用自定义的数据结构存储
       :param network: 文件路径
    """
    global node_num, edge_num, graph, seeds
    data_lines = open(network, 'r').readlines()
    seed_lines = open(seed, "r").readlines()
    node_num = int(data_lines[0].split()[0])
    edge_num = int(data_lines[0].split()[1])

    for data_line in data_lines[1:]:
        start, end, weight = data_line.split()
        graph.add_edge(int(start), int(end), float(weight))

    for seed_line in seed_lines:
        seeds.append(int(seed_line))


def IC():
    """
    实现独立级联模型。
    节点尝试激活它的所有邻居（每个只尝试激活一次），然后新激活的节点再尝试激活它们的邻居，
    重复该过程直到没有节点再可以被激活。
    """
    # print('seeds',seeds)
    count = len(seeds)
    activity_set = set(seeds)
    active_nodes = set(seeds)
    while activity_set:
        new_activity_set = set()
        for seed in activity_set:
            for node, weight in graph.get_neighbors(seed):
                if node not in active_nodes:
                    if random.random() < weight:
                        active_nodes.add(node)
                        new_activity_set.add(node)
        count += len(new_activity_set)
        activity_set = new_activity_set
    return count


def LT():
    """
    实现线性阈值模型：
    节点周围的权值之和大于该节点的阈值，则该节点被激活。
    """
    count = len(seeds)
    activity_set = set(seeds)
    active_nodes = set(seeds)
    node_threshold = {}
    node_weights = {}
    while activity_set:
        new_activity_set = set()
        for seed in activity_set:
            for node, weight in graph.get_neighbors(seed):
                if node not in active_nodes:
                    if node not in node_threshold:
                        node_threshold[node] = random.random()
                        node_weights[node] = 0
                    node_weights[node] += weight
                    if node_weights[node] >= node_threshold[node]:
                        active_nodes.add(node)
                        new_activity_set.add(node)
        count += len(new_activity_set)
        activity_set = new_activity_set
    return count


def calculate_influence(Sk, model_type, _graph):
    global seeds, worker, model, graph
    graph = _graph
    model = model_type
    seeds = Sk
    worker = []
    worker_num = 8
    # print('S',seeds)
    # print('模型',model)

    create_worker(worker_num, int(10000 / worker_num))
    result = []
    for w in worker:
        # print(w.outQ.get())
        result.append(w.outQ.get())
    # print('%.2f' % (sum(result) / 10000))
    finish_worker()
    return sum(result) / 10000


model = 'IC'
seeds = []
graph = Graph()
pGraph = pGraph()

if __name__ == "__main__":
    """
    define global variables:
    node_num: total number of nodes in the network
    edge_num: total number of edges in the network
    graph: represents the network
    seeds: the list of seeds
    """
    node_num = 0
    edge_num = 0
    seeds = []
    model = 'IC'
    """
    command line parameters
    """
    network_path = "test_data/NetHEPT.txt"
    seed_path = "test_data/seeds2.txt"

    termination = 10
    start = time.time()

    read_file(network_path, seed_path)

    worker = []
    worker_num = 2
    create_worker(worker_num, int(10000 / worker_num))
    result = []
    for w in worker:
        # print(w.outQ.get())
        result.append(w.outQ.get())
    print('%.2f' % (sum(result) / 10000))
    finish_worker()
    end = time.time()
    print(end - start)
