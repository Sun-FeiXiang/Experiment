import csv
import numpy as np
import random
import copy


class IC:
    __K = 5  # 循环次数
    __N = 0  # 元素个数
    __start = 0  # 初始激活节点，默认为0

    # 路径，初始集合
    def __init__(self, filePath, A):
        self.filePath = filePath
        self.A = A
        networks = self.loadNetFromCSV()
        self.spread(networks)

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
        # 开始循环
        for i in range(K):  # 对比实验
            state = old_state.copy()
            one_state = state.copy()
            one_activated = copy.copy(activated)  # 每一次的激活节点
            cur_activated = copy.copy(one_activated)

            p_vw = np.zeros((N, N))  # 节点被激活时，激活其它节点的概率,a对b的影响等于b对a的影响
            for random_i in range(N):
                for random_j in range(random_i + 1, N):
                    if networks[random_i][random_j] == 1:
                        num = random.random()
                        p_vw[random_i][random_j] = num
                        p_vw[random_j][random_i] = num
            print("开始循环")
            # j = 1
            activated_num = 0
            while True:  # 激活节点的个数不再发生改变
                # print("开始第",j,"次传播")
                # j = j + 1
                # 遍历已激活节点
                for m in cur_activated:  #
                    # 遍历领接矩阵的m行
                    for n in range(N):
                        # 与激活节点相连接的节点集合，节点m第一次尝试激活n
                        if networks[m][n] == 1:
                            activated_num = activated_num + 1
                            if p_vw[m][n] > 0:
                                # 节点activated[m]第一次尝试激活节点n
                                if one_state[n] == 0:  # 存在当前节点未激活n，有可能存在其它节点激活了n
                                    # 判断是否被激活
                                    try_activate = random.random() + 0.1
                                    # print(try_activate,"?",p_vw[m][n])
                                    if try_activate > p_vw[m][n]:
                                        one_state[n] = 1
                                        one_activated.append(n)
                                        p_vw[m][n] = -1
                                        # print(m, "激活", n, "成功")
                                        break
                                    else:
                                        p_vw[m][n] = -1  # 每个节点只有一次激活其邻居的机会，尝试激活后，不管激活是否成功，m都不能再尝试激活n
                                        # print(m,"激活",n,"失败")
                                elif one_state[n] == 1:  # n点已经被激活
                                    p_vw[m][n] = -1
                                    # print(n,'点已经被激活')

                if len(cur_activated) == len(one_activated):  # 激活次数等于总的边数
                    break
                cur_activated = copy.copy(one_activated)  # 更新当前的激活节点

                # c[i][j + 1] = one_state.copy()
                # activated = one_activated.copy()
                # activated_num += len(one_activated)
            print("第", i + 1, "次循环激活节点一共为：", len(one_activated), "个")


if __name__ == '__main__':
    ic = IC('../data/links.csv', [1])

    # networks = np.array([[0,1,0,1],[1,0,1,1],[0,1,0,1],[1,1,1,0]])
    # ic.spread()
