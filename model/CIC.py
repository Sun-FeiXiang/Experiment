"""
基于IC模型，每个节点都有三个状态：非活动态（inactive）、P活动态（P-active）和N活动态（N-active）。
处于非活动状态的个体不受影响。处于P（N）活动状态的个体表示接受正面（负面）影响。正面和负面影响的传播过程与IC模型一样独立展开。
当一个人同时受到正面和负面意见的影响时，负面影响胜于正面影响。
"""

import csv
import numpy as np
import random
import copy

class CIC:
    __K = 5  # 循环次数
    __N = 0  # 元素个数
    __start = 0  # 初始激活节点，默认为0

    # 路径，初始集合
    def __init__(self, filePath, A):
        self.filePath = filePath
        self.A = A
        networks = self.loadNetFromCSV()
        print(networks)
        #self.spread(networks)

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


if __name__ == '__main__':
    ic = CIC('../data/links.csv', [1])