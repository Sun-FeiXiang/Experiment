"""
算法：暴力法求解
     组合出所有的情况，最大的作为解
"""
import networkx as nx

from algorithm.IC.IC import avgIC_cover_size
from algorithm.IC.degreeDiscount import degreeDiscountIC
from itertools import combinations
from algorithm.Spread.Networkx_spread import spread_run_IC

if __name__ == '__main__':
    import time

    start = time.time()

    # 读取网络图
    G = nx.DiGraph()
    with open('../../data/karate_club.edgelist') as f:
        n, m = f.readline().split()
        for line in f:
            u, v = map(int, line.split())
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u, v, weight=1)
            # G.add_edge(u, v, weight=1)
    print('Built graph G')
    print(time.time() - start)

    # 生成固定的传播概率
    from generation.generation_propagation_probability import weight_probability_inEdge
    weight_probability_inEdge(G)

    seed_size = 5
    nodes = G.nodes()
    C = combinations(nodes, seed_size)  # 排列组合

    spread = dict()
    for candidate in C:
        #print(candidate)
        #time2spread = time.time()
        spread[candidate] = spread_run_IC(candidate, G, 1000)
        #print(spread[candidate], '花费时间：', time.time() - time2spread)

    S, val = max(spread.items())
    print('S (by brute-force):', S, ' -->', val)

    # S2 = degreeDiscountIC(G, seed_size, p)
    # print('S (by degree discount):', tuple(S2), ' -->', avgIC_cover_size(G, S2, p, 1000))
    # print('S (by degree discount) spreads to %s nodes (according to brute-force)' % (spread[tuple(sorted(S2))]))
    # print('Total time:', time.time() - start)
    #
    # console = []
