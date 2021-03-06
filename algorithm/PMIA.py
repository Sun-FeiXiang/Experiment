"""
PMIA算法实现
来源：Scalable Influence Maximization for Prevalent Viral Marketing in Large-Scale Social Networks.
"""
from diffusion.Networkx_diffusion import spread_run_IC,spread_run_LT
import networkx as nx
import math
import time
from dataPreprocessing.read_txt_nx import read_Graph
from dataPreprocessing.generation_propagation_probability import fixed_probability

def updateAP(ap, S, PMIIAv, PMIIA_MIPv, Ep):
    ''' Assumption: PMIIAv is a directed tree, which is a subgraph of general G.
    PMIIA_MIPv -- dictionary of MIP from nodes in PMIIA
    PMIIAv is rooted at v.
    '''
    # going from leaves to root
    sorted_MIPs = sorted(PMIIA_MIPv.items(), key=lambda x: len(x[1]), reverse=True)
    for u, _ in sorted_MIPs:
        if u in S:
            ap[(u, PMIIAv)] = 1
        elif not PMIIAv.in_edges([u]):
            ap[(u, PMIIAv)] = 0
        else:
            in_edges = PMIIAv.in_edges([u], data=True)
            prod = 1
            for w, _, edata in in_edges:
                # p = (1 - (1 - Ep[(w, u)])**edata["weight"])
                p = Ep[(w, u)]
                prod *= 1 - ap[(w, PMIIAv)] * p
            ap[(u, PMIIAv)] = 1 - prod


def updateAlpha(alpha, v, S, PMIIAv, PMIIA_MIPv, Ep, ap):
    """
    更新线性系数alpha
    :param alpha:原alpha值
    :param v:节点v
    :param S:种子集合
    :param PMIIAv:节点v去除前缀的最大影响树
    :param PMIIA_MIPv:节点v去除前缀的最大影响路径
    :param Ep:传播概率
    :param ap:边传播概率
    :return:
    """
    # 从根到叶子排序
    sorted_MIPs = sorted(PMIIA_MIPv.items(), key=lambda x: len(x[1]))
    for u, mip in sorted_MIPs:
        if u == v:
            alpha[(PMIIAv, u)] = 1
        else:
            out_edges = PMIIAv.out_edges([u])
            assert len(out_edges) == 1, "node u=%s must have exactly one neighbor, got %s instead" % (u, len(out_edges))
            w = out_edges[0][1]
            if w in S:
                alpha[(PMIIAv, u)] = 0
            else:
                in_edges = PMIIAv.in_edges([w], data=True)
                prod = 1
                for up, _, edata in in_edges:
                    if up != u:
                        # pp_upw = 1 - (1 - Ep[(up, w)])**edata["weight"]
                        pp_upw = Ep[(up, w)]
                        prod *= (1 - ap[(up, PMIIAv)] * pp_upw)
                # alpha[(PMIIAv, u)] = alpha[(PMIIAv, w)]*(1 - (1 - Ep[(u,w)])**PMIIAv[u][w]["weight"])*prod
                alpha[(PMIIAv, u)] = alpha[(PMIIAv, w)] * (Ep[(u, w)]) * prod


def computePMIOA(G, u, theta, S, Ep):
    """
     计算PMIOA -- 以u为根的子图。最大影响输出树。
     使用Dijkstra算法直到路径长度不超过-log（θ）或者没有更多的节点可以到达。
    """
    # 初始化PMIOA
    PMIOA = nx.DiGraph()
    PMIOA.add_node(u)
    PMIOA_MIP = {u: [u]}  # MIP(u,v) for v in PMIOA

    crossing_edges = set([out_edge for out_edge in G.out_edges([u]) if out_edge[1] not in S + [u]])
    edge_weights = dict()
    dist = {u: 0}  # 从根u开始的最短路径

    # grow PMIOA
    while crossing_edges:
        # Dijkstra贪婪准则
        min_dist = float("Inf")
        min_edge = tuple()
        sorted_crossing_edges = sorted(crossing_edges)  # to break ties consistently
        for edge in sorted_crossing_edges:
            if edge not in edge_weights:
                # edge_weights[edge] = -math.log(1 - (1 - Ep[edge])**G[edge[0]][edge[1]]["weight"])
                edge_weights[edge] = -math.log(Ep[edge])
            edge_weight = edge_weights[edge]
            if dist[edge[0]] + edge_weight < min_dist:
                min_dist = dist[edge[0]] + edge_weight
                min_edge = edge
        # check stopping criteria
        if min_dist < -math.log(theta):
            dist[min_edge[1]] = min_dist
            # PMIOA.add_edge(min_edge[0], min_edge[1], {"weight": G[min_edge[0]][min_edge[1]]["weight"]})
            PMIOA.add_edge(min_edge[0], min_edge[1])
            PMIOA_MIP[min_edge[1]] = PMIOA_MIP[min_edge[0]] + [min_edge[1]]
            # update crossing edges
            crossing_edges.difference_update(G.in_edges(min_edge[1]))
            crossing_edges.update([out_edge for out_edge in G.out_edges(min_edge[1])
                                   if (out_edge[1] not in PMIOA) and (out_edge[1] not in S)])
        else:
            break
    return PMIOA, PMIOA_MIP


def updateIS(IS, S, u, PMIOA, PMIIA):
    """
    更新阻塞点
    :param IS:
    :param S:
    :param u:
    :param PMIOA:
    :param PMIIA:
    :return:
    """
    for v in PMIOA[u]:
        for si in S:
            # 如果种子节点是有效的并且被u阻塞，那么它就变得无效
            if (si in PMIIA[v]) and (si not in IS[v]) and (u in PMIIA[v][si]):
                IS[v].append(si)


def computePMIIA(G, ISv, v, theta, S, Ep):
    """
    节点v排除前缀的最大影响路径。。。
    我们从v遍历向内的边开始一个Dijkstra算法。
    每当Dijkstra算法遇到一个种子节点S时，
    它就停止这个分支，不再继续到S的内邻居。
    Dijkstra算法完成后，我们从计算的内树中移除所有节点IS(v，S)。
    :param G:
    :param ISv:无效种子集，在之前选择种子的去除前缀的最大影响路径上
    :param v:
    :param theta:
    :param S:
    :param Ep:
    :return:
    """
    # 初始化PMIIA
    PMIIA = nx.DiGraph()  # 排除前缀的最大影响图
    PMIIA.add_node(v)
    PMIIA_MIP = {v: [v]}  # MIP(u,v) for u in PMIIA  该节点排除前缀的最大影响树

    # 指向节点v的边，且起始点不是v的阻塞点的集合
    crossing_edges = set([in_edge for in_edge in G.in_edges([v]) if in_edge[0] not in ISv + [v]])
    edge_weights = dict()
    dist = {v: 0}  # 从根u开始的最短路径

    # 增长PMIIA
    while crossing_edges:
        # Dijkstra贪婪准则
        min_dist = float("Inf")  # 正无穷
        min_edge = tuple()
        sorted_crossing_edges = sorted(crossing_edges)  # to break ties consistently
        for edge in sorted_crossing_edges:
            # print('edge',edge)
            if edge not in edge_weights:
                # edge_weights[edge] = -math.log(1 - (1 - Ep[edge])**G[edge[0]][edge[1]]["weight"])
                edge_weights[edge] = -math.log(Ep[edge])
            edge_weight = edge_weights[edge]
            if dist[edge[1]] + edge_weight < min_dist:
                min_dist = dist[edge[1]] + edge_weight
                min_edge = edge
        # 检查停止标准
        # print(min_edge, ':', min_dist, '-->', -math.log(theta))
        if min_dist < -math.log(theta):
            dist[min_edge[0]] = min_dist
            # PMIIA.add_edge(min_edge[0], min_edge[1], {"weight": G[min_edge[0]][min_edge[1]]["weight"]})
            PMIIA.add_edge(min_edge[0], min_edge[1])
            PMIIA_MIP[min_edge[0]] = PMIIA_MIP[min_edge[1]] + [min_edge[0]]
            # 更新crossing edges
            crossing_edges.difference_update(G.out_edges(min_edge[0]))
            if min_edge[0] not in S:
                crossing_edges.update([in_edge for in_edge in G.in_edges(min_edge[0])
                                       if (in_edge[0] not in PMIIA) and (in_edge[0] not in ISv)])
        else:
            break
    return PMIIA, PMIIA_MIP


def PMIA(G, k, theta, Ep):
    start_time = time.time()
    # 初始化
    S, timelapse = [], []
    IncInf = dict(zip(G.nodes(), [0] * len(G)))  # 增量影响传播
    PMIIA = dict()  # node to tree
    PMIOA = dict()
    PMIIA_MIP = dict()  # node to MIPs (dict)
    PMIOA_MIP = dict()
    ap = dict()
    alpha = dict()
    IS = dict()
    for v in G:
        IS[v] = []
        PMIIA, PMIIA_MIP = computePMIIA(G, IS[v], v, theta, S, Ep)#最大影响图核最大影响树
        for u in PMIIA[v]:
            ap[(u, PMIIA[v])] = 0  # u在PMIIA[v]中被激活的概率，初始化
        updateAlpha(alpha, v, S, PMIIA[v], PMIIA_MIP, Ep, ap)  # 更新ap
        for u in PMIIA[v]:
            IncInf[u] += alpha[(PMIIA[v], u)] * (1 - ap[(u, PMIIA[v])])
    # print('初始化完成')
    # print('初始化时间:',time.time() - start)
    # 主循环
    for i in range(k):
        u, _ = max(IncInf.items())
        # print(i+1, "node:", u, "-->", IncInf[u])
        IncInf.pop(u)  # 为下一次迭代pop节点u
        PMIOA[u], PMIOA_MIP[u] = computePMIOA(G, u, theta, S, Ep)
        for v in PMIOA[u]:
            for w in PMIIA[v]:
                if w not in S + [u]:
                    IncInf[w] -= alpha[(PMIIA[v], w)] * (1 - ap[(w, PMIIA[v])])
        updateIS(IS, S, u, PMIOA_MIP, PMIIA_MIP)
        S.append(u)

        for v in PMIOA[u]:
            if v != u:
                PMIIA[v], PMIIA_MIP[v] = computePMIIA(G, IS[v], v, theta, S, Ep)
                updateAP(ap, S, PMIIA, PMIIA_MIP, Ep)
                updateAlpha(alpha, v, S, PMIIA[v], PMIIA_MIP, Ep, ap)
                # 添加增量影响
                for w in PMIIA[v]:
                    if w not in S:
                        IncInf[w] += alpha[(PMIIA[v], w)] * (1 - ap[(w, PMIIA[v])])
        timelapse.append(time.time()-start_time)

    return S


if __name__ == "__main__":
    start = time.time()
    G = read_Graph("../data/graphdata/hep.txt")
    G = G.to_directed()#转换为有向图
    read_time = time.time()
    print('读取网络时间：', read_time - start)
    p = 0.01
    theta = 1/320
    Ep = fixed_probability(G,p)
    algorithm_output = PMIA(G, 50,theta,Ep)

    list_IC_hep = []
    for k in range(1, 51):
        S = algorithm_output[0][:k]
        cur_spread = spread_run_IC(G, S, p, 1000)
        cal_time = algorithm_output[1][k - 1]
        print('CCA算法运行时间：', cal_time)
        print('k = ', k, '选取节点集为：', S)
        print('k=', k, '平均覆盖大小：', cur_spread)
        list_IC_hep.append({
            'k': k,
            'run time': cal_time,
            'average cover size': cur_spread,
            'S': S
        })
    import pandas as pd

    df_IC_hep = pd.DataFrame(list_IC_hep)
    df_IC_hep.to_csv('../../data/output/PMIA/IC_PMIA(p=0.01)_hep_Graph.csv')
    print('文件输出完毕——结束')
