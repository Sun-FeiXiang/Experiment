import math


def LogInfo(title='', val=0):
    if title != '':
        print(title, ':', val)
    else:
        print(val)


#   Math pow2
def pow2(t):
    return t * t


#   Math log2
def log2(n):
    return math.log(n) / math.log(2)


#   logc_n^k = \sum_{i=1}^{k} log((n-k+i)/i) = log(n-k+1)+log((n-k+2)/2)+...+log(n/k)
def logcnk(n, k):
    k = min(k, n - k)
    res = 0.0
    for i in range(1, k + 1):
        res = res + math.log((n - k + i) / i)
    return res


#   根据节点的权重生成概率
def gen_random_node_by_weight(numV, pAccumWeight):
    minIdx = 0
    maxIdx = numV - 1

    return maxIdx


#   对于LT级联模型，根据概率的权重生成一个节点
def gen_random_node_by_weight_LT(edges):
    minIdx = 0
    maxIdx = edges.size() - 1

    return maxIdx


#   将概率规范化为累积格式,[0.2, 0.5, 0.3]->[0.2, 0.7, 1.0]
def to_normal_accum_prob(vecGraph):
    print('to_normal_accum_prob')


#   将权重规格化为累积格式
def to_normal_accum_weight(pWeight, numV):
    print('to_normal_accum_weight')
