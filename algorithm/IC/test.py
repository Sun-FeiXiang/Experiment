"""
测试不同文件的文件
"""
from timeit import default_timer as timer

if __name__ == '__main__':
    import time

    start = time.time()
    from dataPreprocessing.read_gpickle_nx import read_gpickle_DiGraph

    G = read_gpickle_DiGraph("../../data/graphs/hep.gpickle")
    read_time = time.time()
    print('读取网络时间：', read_time - start)
    # 生成固定的传播概率
    from dataPreprocessing.generation_propagation_probability import fixed_probability
    Ep = fixed_probability(G, 0.01)
    I = 1000

    from algorithm.greedy.newGreedyIC import newGreedyIC

    list_IC_hep = []
    temp_time = timer()
    for k in range(5, 31, 5):
        S = newGreedyIC(G, k,Ep)
        cal_time = timer() - temp_time
        print('算法运行时间：', cal_time)
        print('选取节点集为：', S)

        from algorithm.IC.IC import avgIC_cover_size

        average_cover_size = avgIC_cover_size(G, S, 0.01, I)
        print('平均覆盖大小：', average_cover_size)

        list_IC_hep.append({
            'k': k,
            'run time': cal_time,
            'average cover size': average_cover_size,
            'S': S
        })
        temp_time = timer()  # 记录当前时间

    import pandas as pd
    df_IC_hep = pd.DataFrame(list_IC_hep)
    df_IC_hep.to_csv('../../data/output/IC_newGreedyIC_hep_init.csv')
    print('文件输出完毕——结束')