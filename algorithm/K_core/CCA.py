from algorithm.K_core.k_core_subgraph import find_kcores
from timeit import default_timer as timer

def mark_overlay(G, node, CO_v, d=1):
    """
    使用bfs覆盖
    :param G: networkx对象
    :param node: 开始节点
    :param d: 度
    :param CO_v:访问标识
    :return: 无，只需将某个节点设置为访问过即可
    """
    q = []  # 队列
    q.append(node)
    level = 0  # 覆盖第几层
    while len(q) > 0 and level < d:
        v = q.pop(0)  # 弹出第一个节点
        G_adj = G.adj[node]
        for key, value in G_adj.items():
            if not CO_v[key]:
                CO_v[key] = True
                q.append(key)
        level = level + 1  # 访问一层


def CCA(G, k, p=0.01, d=1):
    """
    :param G: networkx图对象
    :param k: 初始节点集的节点个数
    :param p: 传播概率
    :param d: 将距离为d的节点标记为覆盖
    :return: 选择的k个点的集合
    """
    S = []
    highest_kcore, k_cores = find_kcores(G)  # 返回最大的k_s值及k_cores分数[(k_s,[(node,degree),...]),...]
    CO_v = dict()  # 节点覆盖属性
    for node in G.nodes:
        CO_v[node] = False

    choose_Num = 0  # 选择的节点数
    for k_cores_line in k_cores:
        key = k_cores_line[0]
        for k_cores_line_one in k_cores_line[1]:
            node = k_cores_line_one[0]
            node_degree = k_cores_line_one[1]
            if choose_Num == k:
                break
            if not CO_v[node]:
                S.append(node)
                # 标记
                mark_overlay(G, node, CO_v)
                choose_Num = choose_Num + 1
        if choose_Num == k:
            break
    return S

def sumTrue(CO_v):
    num = 0
    for key,value in CO_v.items():
        if value == True:
            num = num + 1
    return num


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
        S = CCA(G, k)
        cal_time = timer() - temp_time
        print('算法运行时间：', cal_time)
        print('选取节点集为：', S)

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

    df_IC_random_hep = pd.DataFrame(list_IC_random_hep)
    df_IC_random_hep.to_csv('../../data/output/IC_CCA_hep.csv')
    print('文件输出完毕——结束')
