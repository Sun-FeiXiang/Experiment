# 优先选择度大的节点 使用igraph实现
from igraph import *
from algorithm.priorityQueue import PriorityQueue as PQ
import time
from diffusion import IC


def degree(g, k, p=0.01, mc=1000):
    S, spread, timelapse, start_time = [], [], [], time.time()
    pq = PQ()
    nodes_degree = g.degree()
    nodes = g.vs.indices
    for node, node_degree in zip(nodes, nodes_degree):
        pq.add_task(node, -node_degree)

    for _ in range(k):
        node, node_degree = pq.pop_item()
        print(node,-node_degree)
        S.append(node)
        # Add estimated spread and elapsed time
        cur_spread = IC(g, S, p, mc)
        print(S,':',cur_spread)
        spread.append(cur_spread)
        timelapse.append(time.time() - start_time)

    return (S, spread, timelapse)


if __name__ == "__main__":
    g = Graph.Read_Ncol("../../data/graphdata/hep.txt",directed=False)
    print(mean(g.degree()))
    # greedy_output = degree(g, 20, p=0.01, mc=1000)
    # print("greedy output: " + str(greedy_output[0]))
