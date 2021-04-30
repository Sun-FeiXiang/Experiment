"""
SIR-Model
易感者会不断变为感染者，而感染者又会不断治愈变成康复者，并且不再被感染
S: Susceptibles，易感者，可能被感染的健康人
I: The Infected，感染者，即患者
R: The Recovered，康复者
β: 病毒传染给健康者的概率
γ: 疾病治愈率
INI: 初始状态下易感者、感染者与治愈者的比例
"""

import networkx as nx
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import random


def SIR(DG, topk, max_iter_num, infect_rate=0.8, remove_rate=0.2):
    CG = DG.to_directed()
    # max_weight = max([e[2]['weight'] for e in CG.edges(data=True)])
    N = CG.number_of_nodes()
    node_state = {node: 0 for node in CG}
    nx.set_node_attributes(CG, values=node_state, name="state")  # 为节点添加属性
    for n in topk:
        CG.nodes[n]['state'] = 1
    random.seed(150)
    for e in CG.edges():
        #if CG[e[0]][e[1]]['weight'] == 1:
        CG[e[0]][e[1]]['prob'] = 0.01 #random.uniform(0, 1)
        # else:
        #     CG[e[0]][e[1]]['prob'] = 1

    all_infect_nodes = []  # 累计受感染节点
    all_infect_nodes.extend(topk)
    all_count_infect = [len(all_infect_nodes)]  # 记录每一次受感染的节点总数

    infected_digraph = nx.DiGraph()
    infected_digraph.add_nodes_from(topk, time=0)

    all_remove_nodes = []  # 累计治愈节点
    all_count_remove = [0]  # 记录每一次免疫的总数

    count_iter_infect = [len(topk)]  # 记录每次新增的感染节点数量
    count_iter_remove = [0]  # 记录每次新增的免疫节点数量
    all_count_suscep = [N - len(topk)]  # 记录每次易感染节点数量
    for i in range(max_iter_num):
        new_infect = []
        new_remove = []
        # t1 = '%s time' % i + ' %s nodes' % len(all_infect_nodes)
        # print(t1) # 当前有多少个节点被感染
        for v in all_infect_nodes:

            if i != 0 and random.uniform(0, 1) < remove_rate:  # 治愈率0.2,
                CG.remove_node(v)  # 该节点具有免疫能力，不再传播，应该从原始图中去除
                # all_infect_nodes.remove(v)#治愈之后，将不会出现该节点，节点失效
                # infected_digraph.node[v]['time'] = -1
                new_remove.append(v)

            else:  # 节点自身没有治愈
                '''
                Twiteer中，一个节点重要主要是这个节点的入度很多。但是用信息传播模型去衡量这个节点的重要性时，应该
                沿着入度的逆时针方向反向传播过去，而不应该根据原始图的出度进行传播
                '''
                for u1, u2 in list(CG.in_edges(v)):
                    if CG.nodes[u1]['state'] == 0:  # 接触的邻居u1是一个健康个体,其中u2==v
                        if CG[u1][u2]['prob'] <= infect_rate:  # 有infect_rate的概率被感染上
                            CG.nodes[u1]['state'] = 1  # 修改节点状态
                            new_infect.append(u1)
                            infected_digraph.add_edge(u2, u1)
                            infected_digraph.nodes[u1]['time'] = i + 1
                    else:
                        infected_digraph.add_edge(u2, u1)  # 邻居节点已经被感染上，则在infect_graph添加一条边
        count_iter_infect.append(len(new_infect))  # 每次新增感染数量
        for i in new_remove:
            all_infect_nodes.remove(i)
        all_infect_nodes.extend(new_infect)  # 添加新增感染的节点,但是在上面运行过程中，有移除的节点
        all_count_infect.append(len(all_infect_nodes))  # 记录每一次截止感染节点总数

        count_iter_remove.append(len(new_remove))  # 每次新增免疫个体数量
        all_remove_nodes.extend(new_remove)  # 添加新增的免疫节点
        all_count_remove.append(len(all_remove_nodes))  # 记录每一次截止免疫节点总数
        suscep = all_count_infect[-1] - all_count_remove[-1]  # 计算易感个体数量
        all_count_suscep.append(suscep)
    result = [all_count_infect, all_count_remove, all_count_suscep, count_iter_infect, count_iter_remove,
              infected_digraph]
    # 每一次受感染的节点总数，每一次免疫的总数，每次易感染节点数量，每次新增的感染节点数量，每次新增的免疫节点数量，感染子图
    return result  # count_iter_infect,count_iter_remove,count_iter_health


if __name__=="__main__":
    G = nx.read_edgelist("../data/graphdata/NetHEPT.txt", nodetype=int)  # 其他数据集使用此方式读取
    for i in G.neighbors(24325):
        print(i)
    result = SIR(G, [24325, 24394], 100)
