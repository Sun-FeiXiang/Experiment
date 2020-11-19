import networkx as nx


def find_kcores(G,k_s=-1):

    k_cores = {}  # 字典
    highest_kcore = 0  # 记录最高的k-core值
    protein_cores = nx.core_number(G)  # 每个顶点的core值

    # group proteins based on the highest k-core in the network where each protein belong
    for protein, k_core in protein_cores.items():
        if highest_kcore < k_core:
            highest_kcore = k_core
        if k_core in k_cores:
            k_cores[k_core].append(protein)
        else:
            k_cores[k_core] = [protein]
    if k_s == -1:
        return highest_kcore, nx.subgraph(G,k_cores[highest_kcore])
    return k_s, nx.subgraph(G,k_cores[k_s])
