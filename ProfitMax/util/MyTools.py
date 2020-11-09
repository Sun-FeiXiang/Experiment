from functools import reduce

class MyTools():
    @staticmethod
    def MakeClass(className,filepath,P_Father = [],p = []):
        """
            自动生成类和getter setter
        :param className: 类名字符串
        :param filepath: 路径字符串，最好绝对路径，如果生成在当前目录就写空字符串
        :param P_Father: 继承的父类和父类参数列表，格式如下
                [
                    ["父类1","父类1第一个参数名或值","父类1第二个参数名或值","父类1第三个参数名或值",……],
                    ["父类2","父类2第一个参数名或值","父类2第二个参数名或值","父类2第三个参数名或值",……]
                ]
        :param p: 类中的属性列表
        :return:
        """
        lines = [
            "\nclass {0}(".format(className)+(reduce(lambda a, b: a[0] + ", " + b[0], P_Father) if P_Father  else "")+"):",
            "\tdef __init__(self):"
        ]

        for superClass in P_Father:
            str = "\t\t{0}.__init__(self".format(superClass[0]) + \
                  ((", " + reduce(lambda a, b: a + ", " + b, superClass[1:])) if superClass[1:] else "") + ")"
            lines.append(str)
            pass

        lines += ["\t\tself.__{0} = None".format(param) for param in p]
        lines += ["\t\tpass", ""]

        for param in p:
            methodLines = []
            methodLines.append("\t@property")
            methodLines.append("\tdef {0}(self):".format(param))
            methodLines.append("\t\treturn self.__{0}".format(param))
            methodLines.append("")
            methodLines.append("\t@{0}.setter".format(param))
            methodLines.append("\tdef {0}(self, value):".format(param))
            methodLines.append("\t\tself.__{0} = value".format(param))
            methodLines.append("")
            lines += methodLines
            pass
        str="+ (", " + reduce(lambda a, b: a + ", " + b, p_属性列表) if p_属性列表 else "") + "
        lines.append("\tdef ToDict(self):" )
        lines.append('\t\tdict1={')
        lines+=['\t\t\t{0}:{1},'.format("'"+p1+"'",'self.__'+p2) for p1,p2 in zip(p,p)]
        lines.append('\t\t\t}\n\t\treturn dict1\n\tpass')
        print(lines)
        lines = [i+"\n" for i in lines]
        with open(filepath,"a+",encoding="utf-8") as fp:
            fp.writelines(lines)
            print("类{0}生成成功".format(className))
            pass
        pass
    pass

MyTools.MakeClass("ResultInfo",__file__,p=["RunningTime", "Profit", "ProfitOrg", "ProfitLowerLattice", "ProfitUpperLattice", "TotalCost",
                 "SeedSize", "SimpleSearchLatticeSize", "LowerLatticeSize", "SearchLatticeSize", "EstNodeSize",
                 "RRsetsSize", "VecSeed", "VecLowerLattice", "VecSearchLattice", "VecProfitBound", "VecInfluence"])

class ResultInfo():
	def __init__(self):
		self.__RunningTime = None
		self.__Profit = None
		self.__ProfitOrg = None
		self.__ProfitLowerLattice = None
		self.__ProfitUpperLattice = None
		self.__TotalCost = None
		self.__SeedSize = None
		self.__SimpleSearchLatticeSize = None
		self.__LowerLatticeSize = None
		self.__SearchLatticeSize = None
		self.__EstNodeSize = None
		self.__RRsetsSize = None
		self.__VecSeed = None
		self.__VecLowerLattice = None
		self.__VecSearchLattice = None
		self.__VecProfitBound = None
		self.__VecInfluence = None
		pass

	@property
	def RunningTime(self):
		return self.__RunningTime

	@RunningTime.setter
	def RunningTime(self, value):
		self.__RunningTime = value

	@property
	def Profit(self):
		return self.__Profit

	@Profit.setter
	def Profit(self, value):
		self.__Profit = value

	@property
	def ProfitOrg(self):
		return self.__ProfitOrg

	@ProfitOrg.setter
	def ProfitOrg(self, value):
		self.__ProfitOrg = value

	@property
	def ProfitLowerLattice(self):
		return self.__ProfitLowerLattice

	@ProfitLowerLattice.setter
	def ProfitLowerLattice(self, value):
		self.__ProfitLowerLattice = value

	@property
	def ProfitUpperLattice(self):
		return self.__ProfitUpperLattice

	@ProfitUpperLattice.setter
	def ProfitUpperLattice(self, value):
		self.__ProfitUpperLattice = value

	@property
	def TotalCost(self):
		return self.__TotalCost

	@TotalCost.setter
	def TotalCost(self, value):
		self.__TotalCost = value

	@property
	def SeedSize(self):
		return self.__SeedSize

	@SeedSize.setter
	def SeedSize(self, value):
		self.__SeedSize = value

	@property
	def SimpleSearchLatticeSize(self):
		return self.__SimpleSearchLatticeSize

	@SimpleSearchLatticeSize.setter
	def SimpleSearchLatticeSize(self, value):
		self.__SimpleSearchLatticeSize = value

	@property
	def LowerLatticeSize(self):
		return self.__LowerLatticeSize

	@LowerLatticeSize.setter
	def LowerLatticeSize(self, value):
		self.__LowerLatticeSize = value

	@property
	def SearchLatticeSize(self):
		return self.__SearchLatticeSize

	@SearchLatticeSize.setter
	def SearchLatticeSize(self, value):
		self.__SearchLatticeSize = value

	@property
	def EstNodeSize(self):
		return self.__EstNodeSize

	@EstNodeSize.setter
	def EstNodeSize(self, value):
		self.__EstNodeSize = value

	@property
	def RRsetsSize(self):
		return self.__RRsetsSize

	@RRsetsSize.setter
	def RRsetsSize(self, value):
		self.__RRsetsSize = value

	@property
	def VecSeed(self):
		return self.__VecSeed

	@VecSeed.setter
	def VecSeed(self, value):
		self.__VecSeed = value

	@property
	def VecLowerLattice(self):
		return self.__VecLowerLattice

	@VecLowerLattice.setter
	def VecLowerLattice(self, value):
		self.__VecLowerLattice = value

	@property
	def VecSearchLattice(self):
		return self.__VecSearchLattice

	@VecSearchLattice.setter
	def VecSearchLattice(self, value):
		self.__VecSearchLattice = value

	@property
	def VecProfitBound(self):
		return self.__VecProfitBound

	@VecProfitBound.setter
	def VecProfitBound(self, value):
		self.__VecProfitBound = value

	@property
	def VecInfluence(self):
		return self.__VecInfluence

	@VecInfluence.setter
	def VecInfluence(self, value):
		self.__VecInfluence = value

	def ToDict(self):
		dict1={
			'RunningTime':self.__RunningTime,
			'Profit':self.__Profit,
			'ProfitOrg':self.__ProfitOrg,
			'ProfitLowerLattice':self.__ProfitLowerLattice,
			'ProfitUpperLattice':self.__ProfitUpperLattice,
			'TotalCost':self.__TotalCost,
			'SeedSize':self.__SeedSize,
			'SimpleSearchLatticeSize':self.__SimpleSearchLatticeSize,
			'LowerLatticeSize':self.__LowerLatticeSize,
			'SearchLatticeSize':self.__SearchLatticeSize,
			'EstNodeSize':self.__EstNodeSize,
			'RRsetsSize':self.__RRsetsSize,
			'VecSeed':self.__VecSeed,
			'VecLowerLattice':self.__VecLowerLattice,
			'VecSearchLattice':self.__VecSearchLattice,
			'VecProfitBound':self.__VecProfitBound,
			'VecInfluence':self.__VecInfluence,
			}
		return dict1
	pass
