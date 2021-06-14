import pandas as pd
from dataPreprocessing.read_txt_nx import read_Graph
from algorithm.test.CBPCA2 import get_E_i,node_core_number,get_node_degree,get_node_h
import math
from algorithm.MCDE import get_node_entropy
from heapdict import heapdict

def get_data_info(file):
    df = pd.read_csv(file)
    node_degree,node_core,node_h,node_influence = dict(),dict(),dict(),dict()
    for row in df.itertuples():
        # print(row['core'], row['degree'],row['h'],row['influence'],row['u'])  # 输出每一行
        u = getattr(row, 'u')
        node_core[u] = getattr(row, 'core')
        node_degree[u] = getattr(row, 'degree')
        node_h[u] = getattr(row, 'h')
        node_influence[u] = getattr(row, 'influence')
    return node_degree,node_core,node_h,node_influence


if __name__=="__main__":
    G = read_Graph("../data/graphdata/DBLP.txt")
    node_degree,node_core,node_h,node_influence = get_data_info("../data/output/DBLP_info.csv")
    total_node_degree = get_node_degree(G)
    total_node_h = get_node_h(G)
    total_node_core = node_core_number(G)
    node_E_i = get_E_i(G, total_node_core)
    node_entropy = get_node_entropy(G, total_node_core)
    node_GI = heapdict()
    for u in G.nodes:
        node_GI[u] = - (node_E_i[u] * math.sqrt(total_node_degree[u]**2+total_node_h[u]**2))

    node_GI_top = dict()
    for i in range(50):
        u,u_GI = node_GI.popitem()
        node_GI_top[u] = u_GI
    count = len(set(node_GI_top.keys()).intersection(set(node_influence.keys())))
    print(count/50)