import itertools
from heapq import *


class PriorityQueue(object):
    def __init__(self):
        self.pq = []  # 堆中排列项的列表
        self.entry_finder = {}  # 任务到条目的映射
        self.REMOVED = '<removed-task>'  # 已删除任务的占位符
        self.counter = itertools.count()  # 唯一序列计数
        # print(self.counter)

    def add_task(self, task, priority=0):
        """添加新任务或更新现有任务的优先级"""
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove_task(self, task):
        """将现有任务标记为REMOVED。如果未找到则Raise KeyError。"""
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_item(self):
        """删除并返回优先级最低的任务。如果为空，则引发键错误。"""
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task, priority
        raise KeyError('pop from an empty priority queue')

    def __str__(self):
        return str([entry for entry in self.pq if entry[2] != self.REMOVED])


if __name__ == "__main__":
    pq = PriorityQueue()
    pq.add_task(pq.REMOVED, -100)
    pq.add_task(1, -75)
    pq.add_task(2, -50)
    pq.add_task(pq.REMOVED, -25)
    print(pq.pop_item())
