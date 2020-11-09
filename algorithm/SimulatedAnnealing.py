# 模拟退火算法(SimulatedAnnealing)是基于Monte-Carlo迭代求解策略的一种随机寻优算法,主要用于组合优化问题的求解。
# f(x)=x^3-60x^2-4x+6
# 启发式搜索算法，即按照预定的控制策略进行搜索，在搜索过程中获取的中间信息将用来改进控制策略

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math


# define aim function
def aimFunction(x):
    y = x ** 3 - 60 * x ** 2 - 4 * x + 6
    return y


x = [i / 10 for i in range(1000)]
y = [0 for i in range(1000)]
for i in range(1000):
    y[i] = aimFunction(x[i])

plt.plot(x, y)
plt.show()

T = 1000  # 初始温度
Tmin = 10  # 最小温度值
x = np.random.uniform(low=0, high=100)  # 初始x值
k = 50  # 内循环次数
y = 0  # 初始结果
t = 0  # 次数
while T >= Tmin:
    for i in range(k):
        # 计算y
        y = aimFunction(x)
        # 利用变换函数在x附近生成一个新的x
        xNew = x + np.random.uniform(low=-0.055, high=0.055) * T
        if 0 <= xNew <= 100:
            yNew = aimFunction(xNew)
            if yNew - y < 0:  # y变小了
                x = xNew
            else:
                # metropolis 准则
                p = math.exp(-(yNew - y) / T)
                r = np.random.uniform(low=0, high=1)
                if r < p:
                    x = xNew
    t += 1
    # print(t)
    T = 1000 / (1 + t)

y = aimFunction(x)
print('x:', x, ',y:', y)
