 
####This folder contains several different types of graphs.####

* hep.txt – 从arXiv.org爬取的一个协同图, 高能物理 - 理论部分，从1991年到2003年。

* phy.txt – A collaboration graph crawled from arXiv.org, Physics section.

* graph30.txt - random graph with 30 vertices and 120 edges (created by networkx.py)

####本文简要介绍了图形文件的数据格式 ####
- 第1行：两个整数–n和m。
n是图中的节点数，所有节点的编号从0到n-1。m为边数，同一对节点之间的多条边分别计数。

- 第2至m+1行：共有m行。每行包含两个整数，表示由一条边连接的两个顶点。
如前所述，如果在同一对节点之间有两个或多个边，则它们的id将以多行形式显示在一起。
Original source: [http://research.microsoft.com/](http://research.microsoft.com/en-us/people/weic/graphdata.zip)
