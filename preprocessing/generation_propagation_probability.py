# 生成传播概率
import random
# 返回固定传播概率
import networkx as nx


def fixed_probability(G, p):
    """
    :param G: 图
    :param p: 固定概率
    :return: 字典，每条边：概率
    """
    fp = dict()
    for edge in G.edges:
        fp[edge] = p
    return fp

# 用权重表示传播概率，固定值
def weight_probability_fixed(G,p=0.01):
    """
    用weight表示传播概率，使用统一的传播概率，默认为0.01
    :param G: 
    :param p: 
    :return: 
    """
    for edge in G.edges:
        G.edges[edge]['weight'] = p
    
# 用权重表示传播概率，1/in_edge
def weight_probability_inEdge(G):
    """
    用weight表示传播概率，生成e的概率为以e为终点，该点的入边
    :param G: 图
    :param p: 固定概率
    :return: 字典，每条边：概率
    """
    for edge in G.edges:
        G.edges[edge]['weight'] = 1 / G.in_degree(edge[1])


# 用p表示传播概率，随机生成
def p_random(G):
    """
    :param G: 图
    :return: 字典，每条边：概率
    """
    for edge in G.edges:
        G.edges[edge]['p'] = random.random()

# 用p表示传播概率，固定概率
def p_fixed(G,p):
    """
    :param G: 图
    :param p: 固定概率
    :return: 字典，每条边：概率
    """
    for edge in G.edges:
        #G.edges[edge]['weight'] = 1
        G.edges[edge]['p'] = p


def p_fixed_with_link(G,p):
    """
    使用边的p属性表示传播概率，且在固定概率上加入传播联系
    :param G:
    :param p:
    :return:
    """
    for edge in G.edges:
        start = edge[0]
        end = edge[1]
        start_adj = list(G.adj[start].keys())
        end_adj = list(G.adj[end].keys())
        intersection = [i for i in start_adj if i in end_adj]  # O(k^2)
        G.edges[edge]['p'] = p + len(intersection)/(len(start_adj)+len(end_adj))

# 用p表示传播概率，概率为1/in_edge
def p_inEdge(G):
    """
    :param G: 图
    :return: 字典，每条边：概率
    """
    for edge in G.edges:
        G.edges[edge]['p'] = 1 / G.in_degree(edge[1])

def fixed_weight(G):
    for edge in G.edges:
        G.edges[edge]['weight'] = 1
def fixed_distance(G):
    for edge in G.edges:
        G.edges[edge]['distance'] = 1
