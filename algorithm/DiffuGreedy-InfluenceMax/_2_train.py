# -*- coding: utf-8 -*-
"""
提取级联中节点参与的摘要特征
从关注者网络和级联中提取节点中心
将边权重添加到与影响相对应的跟随器网络
按级联测量（复制频率和时间差）
"""

import os

import igraph
from igraph import *
import time
import pandas as pd
from datetime import datetime


def remove_duplicates(cascade_nodes, cascade_times):
    """
    一些推特不止被一个人转发仅仅一次，只保留这个人的第一次转发
    """
    duplicates = set([x for x in cascade_nodes if cascade_nodes.count(x) > 1])
    for d in duplicates:
        to_remove = [v for v, b in enumerate(cascade_nodes) if b == d][1:]
        cascade_nodes = [b for v, b in enumerate(cascade_nodes) if v not in to_remove]
        cascade_times = [b for v, b in enumerate(cascade_times) if v not in to_remove]

    return cascade_nodes, cascade_times


def train_graph(g, train_file):
    """
    使用级联更新边的属性和节点特征
    """
    f = open(train_file)
    g.es["Inf"] = 0
    g.es["Dt"] = 0
    # 通过训练级联进行迭代
    idx = 0
    deleted_nodes = []
    for line in f:
        parts = line.replace("\n", "").split(";")
        # print(parts)
        day = int(parts[0])
        cascade_size = len(parts) - 1

        cascade_nodes = map(lambda x: x.split(" ")[0], parts[1:])
        cascade_times = map(lambda x: datetime.strptime(x.split(" ")[1], '%Y-%m-%d-%H:%M:%S'), parts[1:])

        # print(list(cascade_nodes))  # ['781134', '477647',...] 为当前节点集
        # print(list(cascade_times))  # [...,datetime.datetime(2012, 10, 7, 1, 31, 7)]

        # 移除同一个人的相同转发，只保留第一个
        cascade_nodes, cascade_times = remove_duplicates(list(cascade_nodes), cascade_times)
        # print(cascade_nodes)  # ['781134', '477647',...]
        # print(cascade_times)

        cascade_subgraph = Graph()  # 创建传播子图
        for i in range(len(cascade_nodes)):
            if i == (len(cascade_nodes) - 1):
                break

            # 将所有指向j并遵守时间约束的i边添加到级联图中
            for j in range(i + 1, len(cascade_nodes)):
                edge = j
                try:
                    if g.es[edge]["weight"] <= day:  # and g.es[edge]["Type"]=="Follow"):
                        # i影响j
                        cascade_subgraph.add_edge(cascade_nodes[i], cascade_nodes[j])
                        # 给图的边添加属性
                        g.es[edge]["Inf"] += 1
                        g.es[edge]["Dt"] += (cascade_times[j] - cascade_times[i]).total_seconds()
                except:
                    pass
                idx += 1
        if idx % 100 == 0:
            print("-------------------", idx)

    # -----80 deleted nodes
    # print("Number of nodes not found in the graph: ", len(deleted_nodes))
    f.close()
    return g


"""
Main
"""

log = open("Logs\\time_log.txt", "a")

# es边集，vs点集
g = Graph.Read_Ncol("data\\active_network.txt")
print(g.es[2]['weight'])
#print(g.es[1]["weight"])
start = time.time()
g = train_graph(g, "data/train_cascades.txt")
log.write("Training time:" + str(time.time() - start) + "\n")

# ----------- Subset the graph up to day 25 and remove the weight (which is the day)
log.write("Number of edges with the last week:" + str(len(g.es)) + "\n")
for i in range(26, 33):
    g.delete_edges(g.es.select(weight=i))
del g.es["weight"]
log.write("Number of edges without last week:" + str(len(g.es)) + "\n")  # -- 9 mil difference

# ------------ Store the network
g.write_pickle("data\\trained_network.pickle")

# ------------ Compute structural node statistics
start = time.time()
kcores = g.shell_index(mode="IN")
log.write("K_core time:" + str(time.time() - start) + "\n")

# ------------ Store the node features
pd.DataFrame({"Nodes": g.vs["name"],
              "Degree": g.indegree(),
              "Kcores": kcores}).to_csv("data\centralities.csv", index=False)
