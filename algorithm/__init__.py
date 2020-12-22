
import networkx as nx


if __name__ == "__main__":
    import time

    start = time.time()
    G = nx.read_weighted_edgelist("../data/NetHEPT.txt", comments='#', nodetype=int, create_using=nx.Graph())
    print(len(G.edges))
    read_time = time.time()

    print(G[1])