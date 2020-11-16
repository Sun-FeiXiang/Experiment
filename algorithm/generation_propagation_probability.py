

def fixed_probability(G,p):
    """
    :param G: 图
    :param p: 固定概率
    :return: 字典，每条边：概率
    """
    fp = dict()
    for edge in G.edges:
        fp[edge] = p
    return fp