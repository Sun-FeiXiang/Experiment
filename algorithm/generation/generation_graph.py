import networkx as nx
import matplotlib.pyplot as plt

G = nx.karate_club_graph()
nx.draw(G)
nx.write_edgelist(G, "../../data/karate_club.edgelist", data=False)
plt.show()
