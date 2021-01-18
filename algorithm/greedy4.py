"""
看两步
"""
from Spread.Networkx_spread import runIC
from algorithm.Spread.Networkx_spread import spread_run_IC
from algorithm.priorityQueue import PriorityQueue as PQ
from timeit import default_timer as timer


def greedy(G, k, p=0.01, mc=1000):
    S, spread, timelapse, start_time = [], [], [], timer()
    pq = PQ()
    for node in G.nodes():
        for _ in range(100):
            one_spread = runIC(G, [node], p)  # 一步传播的节点
            two_spread = runIC(G, one_spread, p)  # 两步传播的节点
            pq.add_task(node, -len(two_spread)/100)

    for _ in range(k):
        u, u_ts = pq.pop_item()
        S.append(u)
        cur_spread = spread_run_IC(G, S, p, mc)
        spread.append(cur_spread)
        timelapse.append(timer() - start_time)
    return (S, spread, timelapse)


if __name__ == "__main__":
    import time

    start = time.time()
    from algorithm.data_handle.read_Graph_networkx import read_Graph

    G = read_Graph("../data/graphdata/hep.txt")
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    list_IC_random_hep = []
    temp_time = timer()

    greedy_output = greedy(G, 50)
