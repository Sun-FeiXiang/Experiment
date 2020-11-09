

from time import perf_counter as pc
import time

class Timer:

    def __init__(self,__StartTime=pc(), __LastTime='',
                 __EndTime='',__processName='Unnamed'):
        self.__StartTime = __StartTime
        self.__LastTime = __StartTime
        self.__EndTime = __StartTime
        self.__processName = __processName

    #   更新时间
    def refresh_time(self):
        self.__StartTime = pc()
        self.__LastTime = self.__StartTime
        self.__EndTime = self.__StartTime

    #   记录当前时间
    def record_current_time(self):
        self.__LastTime = self.__EndTime
        #time.sleep(2)
        self.__EndTime = pc()

    #   获取距离上一个记录时的操作时间
    def get_operation_time(self):
        self.record_current_time()
        # print('end',self.__EndTime,'\nlast',self.__LastTime,'\nstart',self.__StartTime)
        return self.__EndTime - self.__LastTime

    #   打印距离上一个记录时刻（给定操作名字）的操作时间
    def log_operation_time(self,operationName=''):
        if operationName != '':
            print('操作',operationName,'用时(s)：', self.get_operation_time())
        else:
            print('用时(s)：', self.get_operation_time())

    #   获取从开始的总时间
    def get_total_time(self):
        self.record_current_time()
        return self.__EndTime - self.__StartTime

    #   打印自开始的总时间
    def log_total_time(self):
        print(self.__processName,'花费的总时间为',self.get_total_time(),'s')

    #   打印从开始到上一时刻的总时间
    def log_sub_total_time(self):
        print('从开始到上一记录时刻，',self.__processName,'花费的时间为',self.__EndTime-self.__StartTime,'s')


if __name__ == '__main__':
    tt = Timer()
    tt.log_operation_time()
    tt.record_current_time()
    tt.record_current_time()
    tt.log_sub_total_time()
    tt.log_total_time()