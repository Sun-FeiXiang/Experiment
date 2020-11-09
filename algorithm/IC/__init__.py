# 基于IC模型的算法
import networkx as nx

G = nx.Graph()
G.add_edge(1,2,weight=1)
G.add_edge(2,3,weight=1)
G.add_edge(1,3,weight=1)
print(G[1][3])

