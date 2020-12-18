"""
实现基于强连通分量的启发式算法 无向图
Strongly Connected Components (SCC)
我们首先用概率（1-p）**w从原始图中删除一条边。
然后我们计算这个图的SCC。
然后，对于连通分量中的所有顶点，我们将该组件中的节点数添加到其分数中。
这个过程重复R次以得到一些平均值。
然后选取得分最高的k个节点。

参考:Kempe et al. "Maximizing the spread of influence through a social network" Claim 2.3
"""

from __future__ import division

import random, operator, time, os
from heapq import nlargest
from copy import deepcopy


def SCC_heuristic(G, k, p, R=20):
    """
     Input:
     G -- 无向图(nx.Graph)
     k -- 种子集的大小(int)
     p -- 所有边的传播概率(int)
     R -- 估计节点得分的迭代次数(int)
     Output:
     S -- 种子集合(元组list: 第一个参数是节点, 第二个参数是它的分数)
    """
    scores = dict(zip(G.nodes(), [0] * len(G)))  # 初始化分数
    start = time.time()
    for it in range(R):
        # remove blocked edges from graph G
        E = deepcopy(G)
        edge_rem = [e for e in E.edges() if random.random() < (1 - p) ** (E[e[0]][e[1]]['weight'])]
        E.remove_edges_from(edge_rem)

        # initialize SCC
        SCC = dict()  # each component is reflection os the number of a component to its members
        explored = dict(zip(E.nodes(), [False] * len(E)))
        c = 0
        # perform BFS to discover SCC
        for node in E:
            if not explored[node]:
                c += 1
                explored[node] = True
                SCC[c] = [node]
                component = E[node].keys()
                for neighbor in component:
                    if not explored[neighbor]:
                        explored[neighbor] = True
                        SCC[c].append(neighbor)
                        component.extend(E[neighbor].keys())

        # add score only to top components
        # topSCC = nlargest(k, SCC.iteritems(), key= lambda (dk,dv): len(dv))
        # for (c, component) in topSCC:
        #     print c, len(component)
        #     weighted_score = 1.0/len(component)
        #     for node in component:
        #         scores[node] += weighted_score

        # update scores
        for c in SCC:
            weighted_score = len(SCC[c])  # score is size of a compnonet
            for node in SCC[c]:
                scores[node] += weighted_score
        print(it + 1, time.time() - start)
    S = nlargest(k, scores.iteritems(), key=operator.itemgetter(1))  # select k nodes with top scores
    return S
