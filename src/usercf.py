import pandas as pd
from collections import defaultdict
from operator import itemgetter
import math
import os
import csv


def LoadDataFromCSV(filepath):
    """
    从CSV文件加载数据，格式为 userId, itemId
    :param filepath: CSV文件路径
    :return: 处理后的数据集
    """
    data = pd.read_csv(filepath)
    train = []
    for _, row in data.iterrows():
        user = row['userId']
        item_list = row['itemId'].split(';')
        for item in item_list:
            train.append([user, item])
    return PreProcessData(train)


def PreProcessData(originData):
    """
    将数据处理为用户-视频的映射字典
    :param originData: 原始数据（用户与视频ID对）
    :return: 用户-视频映射字典
    """
    trainData = dict()
    for user, item in originData:
        trainData.setdefault(user, set()).add(item)
    return trainData


class UserCF(object):
    """ 用户基于协同过滤的推荐算法实现 """
    def __init__(self, trainData, similarity="cosine"):
        self._trainData = trainData
        self._similarity = similarity
        self._userSimMatrix = dict()

    def similarity(self):
        # 构建反向索引：item -> users
        item_user = defaultdict(set)
        for user, items in self._trainData.items():
            for item in items:
                item_user[item].add(user)

        # 计算共现矩阵
        for users in item_user.values():
            for u in users:
                for v in users:
                    if u == v: continue
                    self._userSimMatrix.setdefault(u, defaultdict(int))
                    if self._similarity == "cosine":
                        self._userSimMatrix[u][v] += 1
                    else:  # iif
                        self._userSimMatrix[u][v] += 1. / math.log(1 + len(users))

        # 归一化
        for u, related in self._userSimMatrix.items():
            nu = len(self._trainData[u])
            for v, cuv in related.items():
                nv = len(self._trainData[v])
                self._userSimMatrix[u][v] = cuv / math.sqrt(nu * nv)

    def train(self):
        self.similarity()

    def recommend(self, user, N, K):
        """ 推荐物品 """
        recommends = defaultdict(float)
        interacted = self._trainData.get(user, set())
        for v, sim in sorted(self._userSimMatrix.get(user, {}).items(), key=itemgetter(1), reverse=True)[:K]:
            for item in self._trainData[v]:
                if item in interacted: continue
                recommends[item] += sim
        return dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[:N])

    def recommendperson(self, user, N):
        """ 推荐相似用户 """
        sims = self._userSimMatrix.get(user, {})
        return [other for other, _ in sorted(sims.items(), key=itemgetter(1), reverse=True)[:N]]


def save_recommendperson_results(filepath, model, users, N):
    """
    将 recommendperson 结果保存到 CSV 文件
    :param filepath: 输出文件路径
    :param model: 已训练的 UserCF 模型
    :param users: 用户列表
    :param N: 推荐相似用户数量
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['userId', 'similarUsers'])
        for user in users:
            similar = model.recommendperson(user, N)
            writer.writerow([user, ';'.join(map(str, similar))])