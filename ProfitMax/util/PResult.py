class ResultInfo:

    def __init__(self, __RunningTime, __Profit, __ProfitOrg, __ProfitLowerLattice, __ProfitUpperLattice, __TotalCost,
                 __SeedSize, __SimpleSearchLatticeSize, __LowerLatticeSize, __SearchLatticeSize, __EstNodeSize,
                 __RRsetsSize, __VecSeed, __VecLowerLattice, __VecSearchLattice, __VecProfitBound, __VecInfluence):
        self.__RunningTime = __RunningTime
        self.__Profit = __Profit
        self.__Profit = __Profit
        self.__ProfitOrg = __ProfitOrg
        self.__ProfitLowerLattice = __ProfitLowerLattice
        self.__ProfitUpperLattice = __ProfitUpperLattice
        self.__TotalCost = __TotalCost
        self.__SeedSize = __SeedSize
        self.__SimpleSearchLatticeSize = __SimpleSearchLatticeSize
        self.__LowerLatticeSize = __LowerLatticeSize
        self.__SearchLatticeSize = __SearchLatticeSize
        self.__EstNodeSize = __EstNodeSize
        self.__RRsetsSize = __RRsetsSize
        self.__VecSeed = __VecSeed
        self.__VecLowerLattice = __VecLowerLattice
        self.__VecSearchLattice = __VecSearchLattice
        self.__VecProfitBound = __VecProfitBound
        self.__VecInfluence = __VecInfluence

    def reflesh(self):
        self.__RunningTime = -1.0
        self.__Profit = -1.0
        self.__ProfitOrg = -1.0
        self.__ProfitLowerLattice = 0.0
        self.__ProfitUpperLattice = -1.0
        self.__TotalCost = -1.0
        self.__SeedSize = 0
        self.__SimpleSearchLatticeSize = -1
        self.__LowerLatticeSize = -1
        self.__SearchLatticeSize = -1
        self.__EstNodeSize = 0
        self.__RRsetsSize = -1
        self.__VecSeed.clear()
        self.__VecLowerLattice.clear()
        self.__VecSearchLattice.clear()
        self.__VecProfitBound = []
        self.__VecInfluence = []

    # def get_running_time(self):
    #     return self.__RunningTime

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
        dict1 = {
            'RunningTime': self.__RunningTime,
            'Profit': self.__Profit,
            'ProfitOrg': self.__ProfitOrg,
            'ProfitLowerLattice': self.__ProfitLowerLattice,
            'ProfitUpperLattice': self.__ProfitUpperLattice,
            'TotalCost': self.__TotalCost,
            'SeedSize': self.__SeedSize,
            'SimpleSearchLatticeSize': self.__SimpleSearchLatticeSize,
            'LowerLatticeSize': self.__LowerLatticeSize,
            'SearchLatticeSize': self.__SearchLatticeSize,
            'EstNodeSize': self.__EstNodeSize,
            'RRsetsSize': self.__RRsetsSize,
            'VecSeed': self.__VecSeed,
            'VecLowerLattice': self.__VecLowerLattice,
            'VecSearchLattice': self.__VecSearchLattice,
            'VecProfitBound': self.__VecProfitBound,
            'VecInfluence': self.__VecInfluence,
        }
        return dict1

