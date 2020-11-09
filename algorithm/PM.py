"""

论文：Profit Maximization for Viral Marketing in Online Social Networks:
Algorithms and Analysis

算法：
1.简单贪心启发式算法
beta(S)是所有由S激活节点的总收益
c(S)是选择所有种子节点的总代价
fai(S)=beta(S)-c(S)
目的是找到fai(S)最大时的一个种子集合。
将v加入到种子节点集合S中所获得的边际效益：fai(v|S) = fai(S \cup {v}) - fai(S)
2.双重贪心算法


使用数据集：
Dataset     #Nodes(|V|)     #Edges(|E|)     Type            Avg.degree
Facebook    4k              88k             Undirected      43.7
Wiki-Vote   7 K             104 K           Directed        29.1
Google+     108 K           14 M            Directed        254.1
LiveJournal  5 M            69 M            Directed        28.5
"""
import numpy as np
import networkx as nx
import random


class PM:
    __S_num = 100  # 初始感染集合数目
    __wiki_vote_random_top = 1.0 / 29.1  # wiki_vote激活节点随机数上限

    #  加载网络并且生成传播概率（1/I_i）
    def load_graph(self):
        G = nx.read_adjlist('../data/Wiki-Vote.txt', create_using=nx.DiGraph)
        for u, v, weight in G.edges(data='weight'):
            if G.in_degree[v] == 0:
                G.add_weighted_edges_from([(u, v, 0)])
            else:
                puv = 1.0 / G.in_degree[v]
                G.add_weighted_edges_from([(u, v, puv)])
        return G

    def IC_spread(self, G, random_top):
        #   随机生成__S_num个节点
        #   print(list(G.nodes))
        S = np.random.choice(list(G.nodes), self.__S_num)
        g = nx.DiGraph()
        g.add_nodes_from(S)
        while 1:
            new_activate_num = 0
            for u, v, weight in G.edges(data='weight'):
                activate_probability = random.uniform(0, random_top)
                if u in g.nodes and v not in g.nodes:  # u是激活节点，v不是
                    if activate_probability > weight:
                        g.add_edges_from([(u, v, {'weight': weight})])
                        new_activate_num = new_activate_num + 1
            if new_activate_num == 0:  # 无新节点被激活
                break
        #   抽取子图，将感染图中原本就有的边连接起来
        for u, v, weight in G.edges(data='weight'):
            if (u, v) not in g.edges and u in g.nodes and v in g.nodes:
                g.add_edges_from([(u, v, {'weight': weight})])
        #   print(len(g.nodes))
        return g

    #   生成非uniform的收入和支出，收入使用正态分布bv~N(3.0,1.0);支出使用出度
    def generate_benefits_and_costs(self,g):
        rand_num = np.random.normal(3.0,1.0,len(g.nodes))
        benefits = []
        for i in range(0,len(g.nodes)):
            benefits.append((list(g.nodes)[i],rand_num[i]))
        costs = list(g.out_degree)
        print(benefits,'\n',costs)
        return benefits,costs

    def cal_nodes_independent_profit(self):
        print()

    # Simple Greedy
    def SG(self):
        G = self.load_graph()
        g = self.IC_spread(G, self.__wiki_vote_random_top)
        benefits,costs = self.generate_benefits_and_costs(g)

        print('传播图：', g.in_degree)
        S = []
        total_profit = 0  # 总收益
        while 1:
            profit = 0
            for u in g.nodes:
                if u not in S and profit < self.get_profit_use_SG(g, u, S, total_profit):
                    profit = self.get_profit_use_SG(g, u, S, total_profit)
                    v = u
            if profit <= 0:
                break
            total_profit = total_profit + profit
            S.append(v)
        return S, total_profit

    #    使用简单贪心算法计算利润
    """
        选择该点的收益=选择v和S的收益-选择S的收益
        使用dfs计算收益
    """

    def get_profit_use_SG(self, G, v, S, total_profit):
        cur_profit = 0

        return cur_profit


if __name__ == '__main__':
    pm = PM()
    pm.SG()
