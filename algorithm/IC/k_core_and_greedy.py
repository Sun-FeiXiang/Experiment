"""
测试不同文件的文件
"""
import networkx as nx

from InfluenceMax.IC.IC import runIC
from InfluenceMax.IC.degreeDiscount import degreeDiscountIC

#import matplotlib.pylab as plt
import os
from algorithm.IC.generalGreedy import generalGreedy
from algorithm.IC.IC import runIC

if __name__ == '__main__':
    import time
    start = time.time()

    # read in graph
    G = nx.Graph()
    with open('../../data/graphdata/hep.txt') as f:
        n, m = f.readline().split()
        for line in f:
            u, v = map(int, line.split())
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u,v, weight=1)
            # G.add_edge(u, v, weight=1)
    print('Built graph G')
    # print(G.adj)
    print(time.time() - start)
    print('建立图的时间', time.time() - start)
    k = 30
    alg_start1 = time.time()
    S1 = generalGreedy(G, k)
    print('直接使用算法花费时间',time.time()-alg_start1)
    print('传播图大小',len(runIC(G,S1)))
    # G.remove_edges_from(nx.selfloop_edges(G))
    # alg_start2 = time.time()
    # sub_G = nx.k_core(G)  # 返回k核子图
    # print(sub_G.adj)
    # S2 = generalGreedy(sub_G,k)
    # print('先使用k-core后花费时间',time.time()-alg_start2)
    # print('新传播图大小', len(runIC(G, S2)))
    # #calculate initial set
    # seed_size = 10
    # S = degreeDiscountIC(G, seed_size)
    # print('Initial set of', seed_size, 'nodes chosen')
    # print(time.time() - start)
    #
    # # write results S to file
    # with open('visualisation.txt', 'w') as f:
    #     for node in S:
    #         f.write(str(node) + os.linesep)
    #
    # # calculate average activated set size
    # iterations = 200 # number of iterations
    # avg = 0
    # for i in range(iterations):
    #     T = runIC(G, S)
    #     avg += float(len(T))/iterations
    #     # print i, 'iteration of IC'
    # print('Avg. Targeted', int(round(avg)), 'nodes out of', len(G))
    # print(time.time() - start)
    #
    # with open('IC/lemma1.txt', 'w') as f:
    #     f.write(str(len(S)) + os.linesep)
    #     for node in T:
    #         f.write(str(node) + os.linesep)
    #
    # console = []