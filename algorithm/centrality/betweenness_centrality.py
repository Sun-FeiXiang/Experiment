


def get_hep_betweenness_centrality():
    hep_betweenness_centrality = dict()
    with open("../hep_betweenness_centrality.txt", "r") as f:  # 打开文件
        data = f.read()  # 读取文件
        for line in data.split(', '):
            first = int(line.split(': ')[0])
            second = float(line.split(': ')[1])
            hep_betweenness_centrality[first] = second
    return hep_betweenness_centrality