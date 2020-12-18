def imprecision_func(M, M_eff):
    """
    不精确函数
    :param M:是
    :param M_eff:
    :return:
    """
    epsilon = 1 - M / M_eff
    return epsilon
