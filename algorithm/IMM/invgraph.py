class Graph:
    """
        储存网络图的数据结构
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
        Graph.add_node(self, s)
        Graph.add_node(self, e)
        # add inverse edge
        self.network[e][s] = w

    def get_out_degree(self, source):
        return len(self.network[source])

    def get_neighbors(self, source):
        if source in self.network:
            return self.network[source].items()
        else:
            return []

    def get_neighbors_keys(self, source):
        if source in self.network:
            return self.network[source].keys()
        else:
            return []


