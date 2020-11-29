
import networkx as nx

G = nx.Graph()
G.add_edge(1,2)
G.add_edge(2,3)
G.add_edge(1,6)

print(len(G.adj[1]))
