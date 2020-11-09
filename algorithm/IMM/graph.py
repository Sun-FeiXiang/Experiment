class pGraph:
    """
        存储网络图的数据结构
    """
    def __init__(self):
        self.network = dict()

    def add_node(self, node):
        if node not in self.network:
            self.network[node] = dict()

    """
        添加一条新边
        s ：起始节点
        e ：终止节点
        w ：权重
    """
    def add_edge(self, s, e, w):
        pGraph.add_node(self, s)
        pGraph.add_node(self, e)
        self.network[s][e] = w

    def get_out_degree(self, source):
        return len(self.network[source])

    def get_neighbors(self, source):
        return self.network[source].items()


