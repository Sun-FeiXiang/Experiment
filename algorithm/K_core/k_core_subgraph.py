import networkx as nx


def find_kcores(G):

    k_cores = {}  # 字典
    highest_kcore = 0  # 记录最高的k-core值
    G.remove_edges_from(nx.selfloop_edges(G))
    protein_cores = nx.core_number(G)  # 每个顶点的core值

    for protein, k_core in protein_cores.items():
        if highest_kcore < k_core:
            highest_kcore = k_core
        if k_core in k_cores:
            k_cores[k_core].append({protein:G.out_degree(protein)})
        else:
            k_cores[k_core] = [{protein:G.out_degree(protein)}]

    # 将k_cores的每一行（一层核）按照度排序，度从大到小
    k_cores_sorted = {}
    for key,value_dict in k_cores.items():
        k_cores_sorted_line = {}
        for one_value_dict in value_dict:
            k_cores_sorted_line[list(one_value_dict.keys())[0]] = list(one_value_dict.values())[0]
        k_cores_sorted_line = sorted(k_cores_sorted_line.items(), key=lambda item: item[1],reverse=True)
        k_cores_sorted[key] = k_cores_sorted_line

    # 将k_cores按照key值排序
    k_cores = sorted(k_cores_sorted.items(), reverse=True)
    #print(k_cores)
    return highest_kcore, k_cores
