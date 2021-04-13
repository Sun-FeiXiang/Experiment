
from CBPCA import CBPCA
from diffusion.Networkx_diffusion import spread_run_IC

if __name__=="__main__":
    import time

    start = time.time()
    from dataPreprocessing.read_txt_nx import read_Graph

    G = read_Graph("../../data/graphdata/phy.txt")
    read_time = time.time()
    print('读取网络时间：', read_time - start)

    C = [0.01*i for i in range(1,100)]
    L = [1*i for i in range(1,11)]
    print(C)
    print(L)
    for i in range(5):
        p = 0.01 *(i+1)
        best_spread = 0
        best_c = p * 10
        best_l = 1
        for c in C:
            for l in L:
                algorithm_output = CBPCA(G, 50, p,c,l)
                S = algorithm_output[0]
                cur_spread = spread_run_IC(G, S, p, 1000)
                if cur_spread > best_spread:
                    best_spread = cur_spread
                    best_l = l
                    best_c = c
        print('当p=',p,'时，c=',best_c,'，l=',best_l,'获得最大收益：',best_spread)
    print('end~')