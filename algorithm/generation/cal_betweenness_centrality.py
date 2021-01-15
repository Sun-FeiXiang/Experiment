import networkx as nx


def out_put_betweenness_centrality():
    G = nx.read_weighted_edgelist("../../data/graphdata/phy.txt", comments='#', nodetype=int, create_using=nx.DiGraph())
    bcs = nx.betweenness_centrality(G)
    # print(bcs)
    with open("phy_betweenness_centrality.txt", "w") as f:
        for key in bcs.keys():
            f.write(str(key))
            f.write(" ")
            f.write(str(bcs[key]))
            f.write("\n")


out_put_betweenness_centrality()
print("文件输出完毕！")
