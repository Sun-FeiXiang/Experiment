
import ProfitMax.util.Argument as arg
import ProfitMax.util.GraphBase as GB
from ProfitMax.util.Timer import Timer
import ProfitMax.util.PResult as pr
graphBase = GB.GraphBase()

def main():
    #   随机化生成种子集合
    argument = arg.Argument(_func=2)
    inFileName = '../data/' + argument._dir + '/' + argument._graphName
    print(inFileName)
    if argument._func == 0 or argument._func == 2:
        #   标准化图数据
        graphBase.format_graph()
        if argument._func == 0:
            return 1
    print('The Begin of ', argument._outFileName,'---')

    mainTimer = Timer('main')
    pResult = pr.ResultInfo()



if __name__ == '__main__':
    main()

