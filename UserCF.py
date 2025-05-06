import pandas as pd
from collections import defaultdict
from operator import itemgetter
import math
import time 

def LoadDataFromCSV(filepath):
    """
    从CSV文件加载数据，格式为 user, video_ids
    :param filepath: CSV文件路径
    :return: 处理后的数据集
    """
    # 读取CSV文件
    data = pd.read_csv(filepath)

    train = []

    for _, row in data.iterrows():
        user = row['user']
        # 以分号为分隔符拆分视频ID
        video_ids = row['video_ids'].split(';')
        for item in video_ids:
            train.append([user, item])
            

    # 预处理数据
    return PreProcessData(train)

def PreProcessData(originData):
    """
    将数据处理为用户-视频的映射字典，结构如下：
        {"User1": {VideoID1, VideoID2, VideoID3,...}
         "User2": {VideoID4, VideoID5, VideoID6,...}
         ...
        }
    :param originData: 原始数据（用户与视频ID对）
    :return: 处理后的用户-视频映射字典
    """
    trainData = dict()
    for user, item in originData:
        trainData.setdefault(user, set())
        trainData[user].add(item)
    return trainData

class UserCF(object):
    """ 用户基于协同过滤的推荐算法实现 """
    def __init__(self, trainData, similarity="cosine"):
        """
        初始化UserCF类，设置训练数据和相似度计算方式
        :param trainData: 训练数据，用户-视频字典
        :param similarity: 相似度计算方法（"cosine" 或 "iif"）
        """
        self._trainData = trainData
        self._similarity = similarity
        self._userSimMatrix = dict()  # 用户相似度矩阵

    def similarity(self):
        item_user = dict()
        for user, items in self._trainData.items():
            for item in items:
                item_user.setdefault(item, set())
                item_user[item].add(user)

        for item, users in item_user.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    self._userSimMatrix.setdefault(u, defaultdict(int))
                    if self._similarity == "cosine":
                        self._userSimMatrix[u][v] += 1
                    elif self._similarity == "iif":
                        self._userSimMatrix[u][v] += 1. / math.log(1 + len(users))

        for u, related_user in self._userSimMatrix.items():
            for v, cuv in related_user.items():
                nu = len(self._trainData[u])
                nv = len(self._trainData[v])
                self._userSimMatrix[u][v] = cuv / math.sqrt(nu * nv)

    def recommend(self, user, N, K):
        """
        用户u对物品i的感兴趣程度：
            p(u,i) = ∑WuvRvi
            其中Wuv代表的是u和v之间的相似度， Rvi代表的是用户v对物品i的感兴趣程度，因为采用单一行为的隐反馈数据，所以Rvi=1。
            所以这个表达式的含义是，要计算用户u对物品i的感兴趣程度，则要找到与用户u最相似的K个用户，对于这k个用户喜欢的物品且用户u
            没有反馈的物品，都累加用户u与用户v之间的相似度。
        :param user: 被推荐的用户user
        :param N: 推荐的商品个数
        :param K: 查找的最相似的用户个数
        :return: 按照user对推荐物品的感兴趣程度排序的N个商品
        """
        recommends = dict()
        related_items = self._trainData[user]
        for v, sim in sorted(self._userSimMatrix[user].items(), key=itemgetter(1), reverse=True)[:K]:
            for item in self._trainData[v]:
                if item in related_items:
                    continue
                recommends.setdefault(item, 0.)
                recommends[item] += sim
        return dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[:N])

    def recommendperson(self, user, N):
        """
        推荐与指定用户最相似的N个人
        :param user: 被推荐的用户
        :param N: 推荐的相似用户个数
        :return: 按相似度排序的N个相似用户
        """
        # 获取用户与其他用户的相似度，并按相似度从高到低排序
        similar_users = sorted(self._userSimMatrix[user].items(), key=itemgetter(1), reverse=True)
        
        # 返回相似度最高的N个用户
        return [user for user, sim in similar_users[:N]]

    def train(self):
        self.similarity()

if __name__ == "__main__":
    # 修改为加载CSV文件
    train = LoadDataFromCSV("Data_backup.csv")
    print("train data size: %d" % (len(train)))
    
    time_start = time.time()
    # 创建并训练UserCF模型
    UserCF_model = UserCF(train)
    UserCF_model.train()

    # 计算模型训练时间
    time_end = time.time()
    print(f"模型训练时间: {time_end - time_start:.2f} 秒")

    # 分别对前4个用户进行视频推荐
    time_start = time.time()
    test_users = list(train.keys())[:5]  # 取前4个用户
    for user in test_users:
        print(f"Recommended users similar to {user}: {UserCF_model.recommendperson(user, 3)}")
        print(f"Recommendations for {user}: {UserCF_model.recommend(user, 5, 80)}")

    # 计算推荐时间
    time_end = time.time()
    print(f"推荐时间: {time_end - time_start:.2f} 秒")
