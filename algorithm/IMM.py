import time

import numpy as np
from igraph import *
from collections import Counter
from scipy.special import comb
from tqdm import tqdm
from model.igraph_diffusion import IC

def get_RRS(G, p, l):  # 获取一个反向可达集
    source = np.random.randint(0, l)  # 随机选点ID
    new_nodes, RRS0 = [source], [source]
    while new_nodes:  # 求所选节点的反向可达集
        B = G.neighborhood(vertices=new_nodes, order=1, mode="in", mindist=0)  # 新节点的邻居节点集合
        A = [item for subset in B for item in subset]  # 新节点的邻居节点集合，转化为一个列表
        success = np.random.uniform(0, 1, len(A)) < p
        temp = list(np.extract(success, A))  # 新节点以概率p随机激活邻居节点，每个节点只能激活一次
        RRS = list(set(RRS0 + temp))  # list(set())可以去除相同元素
        new_nodes = list(set(RRS) - set(RRS0))
        RRS0 = RRS[:]
    return (RRS)


def RIS(G, k, p, mc, l, R):  # 选取一次种子集合
    if mc > len(R):
        for _ in tqdm(range(mc - len(R))):
            R.append(get_RRS(G, p, l))
    SEED = []
    T = []
    for _ in range(k):  # 根据贪心策略选取k个节点，使覆盖反向可达集最多
        flat_map = [item for subset in R for item in subset]
        seed = Counter(flat_map).most_common()[0][0]
        t = Counter(flat_map).most_common()[0]
        SEED.append(seed)
        T.append(t)
        R = [rrs for rrs in R if seed not in rrs]
    return SEED


def get_mc(G, k, p, l, Rtemp):  # 求对于网络G，IC模型下，在传播概率为p时，选取k个影响力最大节点，需要生成的反向可达集个数
    LB = 1
    q = 0.1  # 利用贪心算法得到的结果，与最优结果的偏差指数q
    n = int(math.log(l - 1, 2))
    epsilon = math.sqrt(2) * q
    e = math.e
    c = comb(l, k)
    lamb = (2 + 2 / 3 * epsilon) * (math.log(c, e) + math.log(l, e) + math.log(math.log(l, 2), e)) * l / (
                epsilon * epsilon)
    setn = 0
    for i in range(1, n + 1):  # 经估算，传播概率为0.05时，i = 4时求得的反向可达集个数可达到要求，所以直接从4开始循环节省时间
        x = l / (2 ** i)
        setN = int(lamb / x) + 1
        print("测试反向可达集总数：" + str(setN))
        # Rtemp = [get_RRS(G, p, l) for _ in tqdm(range(setN))]   # 耗时，可以继承前一轮循环的列表节省时间，由于时间成本尚可这里没有继承
        for _ in tqdm(range(setN - setn)):
            Rtemp.append(get_RRS(G, p, l))
        print("Rtemp长：" + str(len(Rtemp)))
        print("测试RR sets 创建完成")
        FR = 0
        Rt = Rtemp
        for _ in range(k):  # 根据贪心策略选取k个节点，使覆盖反向可达集最多，并计算k个节点的影响力
            flat_map = [item for subset in Rt for item in subset]
            seed = Counter(flat_map).most_common()[0][0]
            f = Counter(flat_map).most_common()[0][1]
            FR = FR + f
            Rt = [rrs for rrs in Rt if seed not in rrs]
        FR = FR / setN
        print("i等于" + str(i) + "贪心子集影响力:" + str(FR))
        if (l * FR) >= ((1 + epsilon) * x):  # 判断影响力是否能够满足阈值
            LB = l * FR / (1 + epsilon)
            break
        setn = setN
    # 求解需要反向可达集个数num
    a = math.sqrt(math.log(l, e) + math.log(2, e))
    b = math.sqrt((1 - 1 / e) * (math.log(l, e) + math.log(2, e) + math.log(c, e)))
    m = ((1 - 1 / e) * a + b) ** 2
    Lambda = 2 * l * m / (q * q)
    num = int(Lambda / LB) + 1
    return num


if __name__ == '__main__':
    open('../data/graphdata/NetHEPT.txt').read()
    G = Graph.Read_Edgelist('../data/NetHEPT.txt', directed=False)
    L = len(G.vs)

    MC = []
    S = []
    p = 0.01
    list_IC_hep = []
    for i in range(1, 51):
        R = []
        start_time = time.time()
        mc = get_mc(G, i, p, L, R)  # 返回需要生成反向可达集数目
        print("R长：" + str(len(R)))
        print(str(i) + "个节点需要反向可达集子集数:" + str(mc))
        m = RIS(G, i, p, mc, L, R)  # 返回所选种子集合
        S.append(m)
        MC.append(mc)
        cur_spread = IC(G, m, p, 10000)
        list_IC_hep.append({
            'k': i,
            'run time': time.time()-start_time,
            'average cover size': cur_spread,
            'S': m
        })
    print("种子集合" + str(S))
    print("需要反向可达集子集数：" + str(MC))
    import pandas as pd

    df_IC_random_hep = pd.DataFrame(list_IC_hep)
    df_IC_random_hep.to_csv('../data/output/pageRank/IC_IMM(p=0.01)_NetHEPT_Graph.csv')
    print('文件输出完毕——结束')

