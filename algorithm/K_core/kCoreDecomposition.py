from timeit import default_timer as timer
import networkx as nx

"""
来源DiffuGreedy: An Influence Maximization Algorithm Based on Diffusion Cascades对比算法
"""


def kCoreDecomposition(G, k):
    """
    k—core分解
    :param G:
    :param k:
    :return:
    """
    k_cores = {}  # 字典
    highest_kcore = 0  # 记录最高的k-core值
    G.remove_edges_from(nx.selfloop_edges(G))
    protein_cores = nx.core_number(G)  # 每个顶点的core值

    for protein, k_core in protein_cores.items():
        if highest_kcore < k_core:
            highest_kcore = k_core
        if k_core in k_cores:
            k_cores[k_core].append(protein)
        else:
            k_cores[k_core] = [protein]
    k_cores = sorted(k_cores.items(), reverse=True)

    S = []
    # 从k-cores中依次选择最大的点，相同的则顺序选择
    for key, value_list in k_cores:
        for value in value_list:
            if len(S) == k:
                break
            if value not in S:
                S.append(value)
        if len(S) == k:
            break
    return S


if __name__ == "__main__":
    import time

    start = time.time()
    from algorithm.graph_data_handle import read_gpickle

    G = read_gpickle("../../data/graphs/hep.gpickle")
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    # 生成固定的传播概率
    # from algorithm.generation_propagation_probability import fixed_probability
    # Ep = fixed_probability(G, 0.01)

    I = 1000

    list_IC_random_hep = []
    temp_time = timer()
    for k in range(5, 31, 5):
        S = kCoreDecomposition(G, k)
        cal_time = timer() - temp_time
        print('算法运行时间：', cal_time)
        print('k = ', k, '选取节点集为：', S)

        from algorithm.IC.IC import avgIC_cover_size

        average_cover_size = avgIC_cover_size(G, S, 0.01, I)
        print('平均覆盖大小：', average_cover_size)

        list_IC_random_hep.append({
            'k': k,
            'run time': cal_time,
            'average cover size': average_cover_size,
            'S': S
        })
        temp_time = timer()  # 记录当前时间

    import pandas as pd

    # df_IC_random_hep = pd.DataFrame(list_IC_random_hep)
    # df_IC_random_hep.to_csv('../../data/output/IC_CCA_hep.csv')
    # print('文件输出完毕——结束')
