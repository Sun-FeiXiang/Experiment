from diffusion.Networkx_diffusion import spread_run_IC
from heapdict import heapdict


class Node(object):
    def __init__(self, node):
        self.node = node
        self.mg1 = 0
        self.prev_best = None
        self.mg2 = 0
        self.flag = None
        self.list_index = 0


def celfpp(G, k, p=0.1, mc=1000):
    S = set()
    Q = heapdict()
    last_seed = None
    cur_best = None
    node_data_list = []

    for node in G.nodes:
        node_data = Node(node)
        node_data.mg1 = spread_run_IC(G, [node], p, mc)  # 该节点的边际收益
        node_data.prev_best = cur_best
        node_data.mg2 = spread_run_IC(G, [node, cur_best.node], p, mc) if cur_best else node_data.mg1
        node_data.flag = 0
        cur_best = cur_best if cur_best and cur_best.mg1 > node_data.mg1 else node_data
        G.nodes[node]['node_data'] = node_data
        node_data_list.append(node_data)
        node_data.list_index = len(node_data_list) - 1
        Q[node_data.list_index] = - node_data.mg1

    while len(S) < k:
        node_idx, _ = Q.peekitem()
        node_data = node_data_list[node_idx]
        if node_data.flag == len(S):
            S.add(node_data.node)
            del Q[node_idx]
            last_seed = node_data
            continue
        elif node_data.prev_best == last_seed:
            node_data.mg1 = node_data.mg2
        else:
            before = spread_run_IC(G, S, p, mc)
            S.add(node_data.node)
            after = spread_run_IC(G, S, p, mc)
            S.remove(node_data.node)
            node_data.mg1 = after - before
            node_data.prev_best = cur_best
            S.add(cur_best.node)
            before = spread_run_IC(G, S, p, mc)
            S.add(node_data.node)
            after = spread_run_IC(G, S, p, mc)
            S.remove(cur_best.node)
            if node_data.node != cur_best.node: S.remove(node_data.node)
            node_data.mg2 = after - before

        if cur_best and cur_best.mg1 < node_data.mg1:
            cur_best = node_data

        node_data.flag = len(S)
        Q[node_idx] = - node_data.mg1

    return S
