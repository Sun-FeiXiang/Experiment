import csv
import numpy as np
import random
import copy


class LT:
    __K = 5  # 循环次数
    __N = 0  # 元素个数
    __start = 0  # 初始激活节点，默认为0

    # 路径，初始集合
    def __init__(self, filePath, A):
        self.filePath = filePath
        self.A = A
        networks = self.loadNetFromCSV()
        self.spread(networks)

    # 建立一个有向图，测试数据正好构建的是一个无向图
    def loadNetFromCSV(self):
        print("读取数据")
        Nodes = set()
        networks = []
        fileSize = 0  # 文件中字符的行数
        with open(self.filePath, encoding='utf-8') as f:
            data = csv.reader(f)
            for i, line in enumerate(data):  # enumerate 将对象转为索引序列，可以同时获得索引和值。
                # python中节点标号从0开始(连续值)
                Nodes.add(int(line[0]) - 1)
                Nodes.add(int(line[1]) - 1)
                network_line = [int(line[0]) - 1, int(line[1]) - 1]
                networks.append(network_line)
                fileSize = i
        networks = np.array(networks)
        # 存储为邻接矩阵
        N = len(Nodes)  # 节点数，连续的
        self.__N = N
        new_network = np.zeros((N, N), dtype=np.int64)
        for i in range(fileSize):
            a = networks[i][0]
            b = networks[i][1]
            new_network[a][b] = 1
        return new_network

    def spread(self, networks):
        N = networks.shape[0]
        K = self.__K
        A = self.A
        start = self.__start
        # 定义初始状态
        print("定义初始状态")
        state = np.zeros((N), dtype=np.int64)  # 初始化节点激活状态，1代表激活，0代表未激活
        if len(A) == 0:
            state[start] = 1  # 使一个节点成为最初的传播节点
        else:
            # 初始激活节点
            for a in A:
                state[a] = 1
        # 起点备份，保证每次起点相同
        old_state = state.copy()
        activated = A
        # 统计每个节点的邻居个数，每个点给其邻居施加的影响为该点除以其邻居的个数
        neighbors_num = np.zeros((N),dtype=int)
        for neighbors_i in range(N):
            neighbors_num[neighbors_i] = np.sum(networks[neighbors_i]!=0)
        # 开始循环
        for i in range(K):  # 对比实验
            state = old_state.copy()
            one_state = state.copy()
            one_activated = copy.copy(activated)  # 每一次的激活节点
            cur_activated = copy.copy(one_activated)

            threshold = np.zeros((N))  # 生成每个节点的阈值
            for threshold_i in range(N):
                threshold[threshold_i] = random.random()

            weight = np.zeros((N))  # 随机生成影响权重
            for weight_i in range(N):
                weight[weight_i] = random.random()

            print("开始循环")
            while True:
                cur_activated = copy.copy(one_activated)
                for m in range(N):
                    if one_state[m] == 0:   #未被激活节点
                        sum_fluence = 0
                        for n in range(N):
                            if networks[m][n] == 1 and one_state[n] == 1:   #邻居为激活状态的节点给m点施加“压力”
                                sum_fluence += weight[n]/neighbors_num[n]
    
                        if threshold[m] < sum_fluence:  # 邻居对m点施加的影响大于其阈值
                            one_state[m] = 1
                            one_activated.append(m)
                if len(cur_activated) == len(one_activated):
                    break
            print("第", i + 1, "次循环激活节点一共为：", len(one_activated), "个")


if __name__ == '__main__':
    ic = LT('../data/links.csv', [1,3,5,7,9])

    # networks = np.array([[0,1,0,1],[1,0,1,1],[0,1,0,1],[1,1,1,0]])
    # ic.spread()
