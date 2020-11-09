# -*- coding: utf-8 -*-
"""
使用train cascades求影响最大化
"""

import os
from igraph import *
import numpy as np
import time

#   返回种子集合及其传播节点个数
def DiffusionCELF(node_cascades, k=100):
    Q = []
    S = []
    final_cascade = set()

    nid = 0
    msg = 1
    cas = 2
    iteration = 3

    for u in node_cascades.keys():
        temp_l = [u]
        cascades = node_cascades[u]
        index, value = max(enumerate([len(final_cascade.union(set(casc))) for casc in cascades]),
                           key=operator.itemgetter(1))
        temp_l.append(value)  # msg
        temp_l.append(index)  # u最好的级联的下标 cas
        temp_l.append(0)  # iteration
        Q.append(temp_l)
    Q = sorted(Q, key=lambda x: x[1], reverse=True)
    # print(Q)

    # CELF
    while len(S) < k:
        u = Q[0]
        if u[iteration] == len(S):
            if len(S) % 20 == 0:
                print(len(S))
            # 保存新的种子节点
            S.append(u[nid])
            final_cascade = final_cascade.union(node_cascades[u[nid]][u[cas]])
            # 从Q中删除
            Q = [l for l in Q if l[0] != u[nid]]

        else:
            # 更新这个节点
            cascades = node_cascades[u[nid]]
            index, value = max(enumerate([len(final_cascade.union(set(casc))) for casc in cascades]),
                               key=operator.itemgetter(1))
            u[msg] = value
            u[cas] = index
            u[iteration] = len(S)
            Q = sorted(Q, key=lambda x: x[1], reverse=True)

    return S, len(final_cascade)


#   时间复杂度为O(KVC) K是种子集合的大小，V是网络节点数目，C是每个节点边的数目
def DiffusionGreedy(seed_set_cascades, seed_set_size=100):
    #   print(list(seed_set_cascades.items())[:5])
    seed_set = []
    final_cascade = set()
    a = 0
    while len(seed_set) < seed_set_size:
        max_seed = ''
        max_val = 0
        max_idx = np.NaN

        for seed, cascades in seed_set_cascades.items():
            # 边际增益由目前级联和候选级联的联合集给出
            index, value = max(enumerate([len(final_cascade.union(set(casc))) for casc in cascades]),
                               key=operator.itemgetter(1))
            # print('index',index,'\nvalue',value)
            if value > max_val:
                max_val = value
                max_seed = seed
                max_idx = index

        # 选择本步的级联
        chosen_cascade = set(seed_set_cascades[max_seed][max_idx])
        final_cascade = final_cascade.union(chosen_cascade)
        #  添加种子节点并更新种子级联集合（只保留已经未遍历过的）
        seed_set.append(max_seed)
        del seed_set_cascades[max_seed]
        print('第'+a+'次')
        a = a + 1
        print('seed_set',seed_set)
    return seed_set,len(final_cascade)


"""        
Main
"""
node_cascades = {}

print("loading cascades ...")

f = open("data/train_cascades.txt")
idx = 0
for line in f:
    idx += 1
    cascade = line.split(";")

    op_id = cascade[1].split(" ")[0]  # 原始用户id
    cascade = set(map(lambda x: x.split(" ")[0], cascade[2:]))  # 转发用户id

    if op_id in node_cascades:
        node_cascades[op_id].append(cascade)
    else:
        node_cascades[op_id] = []
        node_cascades[op_id].extend(list(cascade))

    if idx % 1000 == 0:
        print("-----------", idx)
# print(node_cascades) # 以字典存储原始id及其转发的id，eg:{..., '820656': ['1500375', '791681'],...}
print("done with cascades, moving to  the algorithm")

start = time.time()
# seed_no = 3000
# S, estimated_spread = DiffusionGreedy(node_cascades,seed_no)

seed_no = 30
S, estimated_spread = DiffusionCELF(node_cascades, seed_no)
print(S,'\n',estimated_spread)
# f = open("Seed sets\\diffusion_celf_seeds.txt", "w")
# for i in S:
#     f.write(str(i))
# f.close()
#
# log = open("Logs\\time_log.txt", "a")
# log.write("\n DiffusionCELF " + str(seed_no) + " :" + str(time.time() - start) + "\n")
# log.close()
#
# k = open("diffu_greedy_estimate.txt", "w")
# k.write(" " + str(estimated_spread))
# k.close()
