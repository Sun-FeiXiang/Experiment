# coding: UTF-8
# 解决图显示中文乱码
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
import matplotlib.pyplot as plt

x = np.array([2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1])
y = np.array([2.4, 0.7, 2.9, 2.2, 3, 2.7, 1.6, 1.1, 1.6, 0.9])

# 求平均值并做中心化处理
mean_x = np.mean(x)
mean_y = np.mean(y)
scaled_x = x - mean_x
scaled_y = y - mean_y
data = np.array([[scaled_x[i], scaled_y[i]] for i in range(len(scaled_x))])

# 求协方差矩阵
cov = np.cov(scaled_x, scaled_y)
# 或者求散度矩阵=(n-1)*cov
sca = (len(scaled_x) - 1) * cov
# 计算特征值和特征向量
eig_val, eig_vec = np.linalg.eig(cov)

# 构建新的空间
new_data = np.transpose(np.dot(eig_vec, np.transpose(data)))

# 选择主要成分
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(len(eig_val))]
eig_pairs.sort(reverse=True)
feature = eig_pairs[0][1]  # 这里只选择一个
# 原数据*选择的特征向量，得出降维的数据
new_data_reduced = np.transpose(np.dot(feature, np.transpose(data)))

# sklearn应用
from sklearn.decomposition import PCA
import numpy as np

pca = PCA(n_components=1)
pca.fit(data)
result = pca.transform(data)

plt.plot(scaled_x, scaled_y, 'o', color='blue', label='原始数据')
plt.plot(new_data[:, 0], new_data[:, 1], '*', color='orange', label='调整后的数据')
# 降维后的数据
plt.plot(new_data_reduced, [1.2] * 10, '>', color='black', label='降维后的数据')
plt.plot(result, [1] * 10, '<', color='black', label='sklearn降维后的数据')

xmin, xmax = scaled_x.min(), scaled_x.max()
ymin, ymax = scaled_y.min(), scaled_y.max()
dx = (xmax - xmin) * 0.2
dy = (ymax - ymin) * 0.2
plt.xlim(xmin - dx, xmax + dx)
plt.ylim(ymin - dy, ymax + dy)
plt.plot([eig_vec[:, 0][0], 0], [eig_vec[:, 0][1], 0], color='red')
plt.plot([eig_vec[:, 1][0], 0], [eig_vec[:, 1][1], 0], color='green')
plt.legend()
plt.show()
