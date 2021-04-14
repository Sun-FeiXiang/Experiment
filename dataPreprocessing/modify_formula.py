import pandas as pd
from dataPreprocessing.read_txt_nx import read_Graph
from algorithm.test.CBPCA import get_E_i,node_core_number
import math
from algorithm.MCDE import get_node_entropy

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
    G = read_Graph("../data/facebook_combined.txt")
    node_degree,node_core,node_h,node_influence = get_data_info("../data/output/facebook_info.csv")
    total_node_core = node_core_number(G)
    node_E_i = get_E_i(G, total_node_core)
    node_entropy = dict()
    node_GI = dict()
    for u in node_influence.keys():
        node_GI[u] = node_E_i[u] * math.sqrt(node_degree[u]**2+node_h[u]**2)
        node_entropy[u] = get_node_entropy(G,total_node_core)
    node_GI = sorted(node_GI.items(),key = lambda x:x[1],reverse=True)
    node_influence = sorted(node_influence.items(),key = lambda x:x[1],reverse=True)
