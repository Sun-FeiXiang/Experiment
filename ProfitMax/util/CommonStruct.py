



class Graph:
    """
    node:{id,value}
    nodeList:[node,node,...]
    edge:(int,float)
    edgeList:[(),(),...]

    """
    def __init__(self,node,nodeList,edge,edgeList):
        self.node = node
        self.nodeList = nodeList
        self.edge = edge
        self.edgeList = edgeList




class Node:

    def __init__(self,id,value):
        self.id = id
        self.value = value

    def small(self,Ele1,Ele2):
        return Ele1.value < Ele2.value

    def greater(self,Ele1,Ele2):
        return Ele1.value > Ele2.value

