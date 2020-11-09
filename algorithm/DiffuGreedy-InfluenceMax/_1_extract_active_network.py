# -*- coding: utf-8 -*-
"""
@author: georg

Extract the user ids of the nodes that are actively retweeting during the time span of
the follower network. Extract the train and test cascades. 
Filter the graph to contain only these nodes.
"""

import os
import time


def split_train_and_test(cascades_file):
    """
    将级联分为录制第25天之前和之后开始的级联.
    保证活跃转发的用户id
    """
    f = open(cascades_file)
    ids = set()  # 原始用户ID集合
    train_cascades = []
    test_cascades = []
    counter = 0

    for line in f:

        date = line.split(" ")[1].split("-")  # 原始发布时间的年月日和具体时间
        original_user_id = line.split(" ")[2]  # 原始用户ID
        retweets = next(f).replace(" \n", "").split(" ")  # 转发用户ID
        # 仅保存2012.9.28到2012.10.29.的级联节点
        if int(date[0]) == 2012:

            retweet_ids = ""
            # 最后7天保存为测试集
            if int(date[1]) == 10 and 23 <= int(date[2]) <= 29:
                ids.add(original_user_id)
                cascade = ""
                for i in range(0, len(retweets) - 1, 2):
                    ids.add(retweets[i])
                    retweet_ids = retweet_ids + " " + retweets[i]
                    cascade = cascade + ";" + retweets[i] + " " + retweets[i + 1]  # 转发ID和时间：eg ...3862
                    # 2012-10-27-18:05:02;...

                # 每个级联也保存了原始用户和时间
                date = str(int(date[2]) + 3)
                op = line.split(" ")
                op = op[2] + " " + op[1]  # 原始用户和时间
                test_cascades.append(date + ";" + op + cascade)  # 日期（几号）；原始用户和时间（1个）+转发用户和时间

            # 剩余天数的数据用于训练
            elif int(date[1]) == 10 or (int(date[1]) == 9 and int(date[2]) >= 28):
                ids.add(original_user_id)
                cascade = ""
                for i in range(0, len(retweets) - 1, 2):
                    ids.add(retweets[i])
                    retweet_ids = retweet_ids + " " + retweets[i]
                    cascade = cascade + ";" + retweets[i] + " " + retweets[i + 1]
                if int(date[1]) == 9:
                    date = str(int(date[2]) - 27)
                else:
                    date = str(int(date[2]) + 3)
                op = line.split(" ")
                op = op[2] + " " + op[1]
                train_cascades.append(date + ";" + op + cascade)

        counter += 1
        if counter % 100000 == 0:
            print("------------" + str(counter))
    f.close()
    # print('train_cascades',train_cascades)
    # print('test_cascades',test_cascades)
    # print('ids',ids)
    return train_cascades, test_cascades, ids


"""
Main
"""
# os.chdir("path\\to\\Data")

#   "a" - 追加 - 会追加到文件的末尾
log = open("Logs\\time_log.txt", "a")

start = time.time()

# 划分原始推特级联
train_cascades, test_cascades, ids = split_train_and_test("Init_Data\\total.txt")
print(train_cascades)
# 保存级联
print("Size of train:", len(train_cascades))
print("Size of test:", len(test_cascades))

# #   新建只写
# with open("data/train_cascades.txt", "w") as f:
#     for cascade in train_cascades:
#         f.write(cascade + "\n")
#
# with open("data/test_cascades.txt", "w") as f:
#     for cascade in test_cascades:
#         f.write(cascade + "\n")
#
# # ------- Keep the processing time
# log.write("Cascade extraction time :" + str(time.time() - start) + "\n")
#
# start = time.time()
#
# # 存储原始用户节点
# f = open("data/active_users.txt", "w")
# for uid in ids:
#     f.write(uid + "\n")
# f.close()
#
# # 保存激活用户子网
# g = open("data/active_network.txt", "w")
#
# f = open("Init_Data\\graph_170w_1month.txt")
#
# found = 0
# idx = 0
# for line in f:
#     edge = line.split(" ")
#
#     if edge[0] in ids and edge[1] in ids:
#         found += 1
#         g.write(line)
#     idx += 1
#     if idx % 1000000 == 0:
#         print(idx)
#
# f.close()
# g.close()
#
# log.write("Filtering of follower graph :" + str(time.time() - start) + "\n")
# log.close()
