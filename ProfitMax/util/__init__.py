# -*- coding: utf-8 -*- #设置中文注释
import igraph as ig

# 创建一个空对象
g = ig.Graph()
# 添加网络中的点
vertex = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
g.add_vertices(vertex)