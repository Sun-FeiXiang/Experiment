
import ProfitMax.util.IOcontroller as ioc
TIO = ioc.IOcontroller()


class Argument:
    """
    参数(15)含义：
    _func: 默认为1，函数参数，0：format graph 1：maximize profit 2: format graph and then maximize profit
        1 -> e: edge file (default), a: adjacent vector,
        2 -> graph only (default, WC cascade model), w: with edge property,
        3 -> r: reverse (default), f: forward, b: bilateral
    _graphName: 图名字，默认为Facebook
    _mode: egr
    _dir:  graphInfo
    _outFileName:  结果名
    _resultFolder: 结果文件夹
    _algName:  算法，默认是双向贪心
    _proDist:  传播模型的概率分布，有WC，TR。默认为WC
    _benefitDist:  收入的类型，默认是uniform
    _costDist:   支出的类型，默认是degree
    _scale:    代价和收入的比率，默认为10
    _para:     代价参数，基本代价的分数，默认是0
    _eps:      IMM/BCT默认允许的错误，默认是0.5
    _numR = 1000000    RR集合生成的数量
    _isIterPrune:  是否使用迭代，默认不使用迭代
    _model:    使用模型，默认是IC
    """
    def __init__(self,_func=1,_graphName='facebook_combined.txt',_mode='egr',
                 _dir='graphInfo',_outFileName='',_resultFolder='result',
                 _algName='doublegreedy',_probDist='WC',_benefitDist='uniform',
                 _costDist='degree',_scale=10.0,_para=0.0,_eps=0.5,_numR=1000000,
                 _isIterPrune=False,_model='IC'):
        self._func = _func
        self._graphName = _graphName
        self._mode = _mode
        self._dir = _dir
        self._outFileName = _outFileName
        self._resultFolder = _resultFolder
        self._algName = _algName
        self._proDist = _probDist
        self._benefitDist = _benefitDist
        self._costDist = _costDist
        self._scale = _scale
        self._para = _para
        self._eps = _eps
        self._numR = _numR
        self._isIterPrune = _isIterPrune
        self._model = _model

        self._outFileName = TIO.get_out_file_name(_graphName, _costDist, _algName, _scale, _para)
        if _isIterPrune:
            _outFileName = "IP"+_outFileName
        if _model == 'LT':
            _outFileName = "LT_"+_outFileName

    def get_outfilename_with_alg(self,algName):
        return TIO.get_out_file_name(self._graphName, self._costDist, self._algName, self._scale, self._para)

    @property
    def func(self):
        return self._func