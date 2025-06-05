# src/userCF.py

import pandas as pd
from collections import defaultdict
from operator import itemgetter
import heapq
import math
import os
import csv


def load_data_from_csv(filepath):
    """
    加载用户-物品交互数据，使用 Pandas 向量化操作高效转换为
    ``{userId: {itemId}}`` 形式。

    Args:
        filepath (str): CSV 文件路径，文件包含 ``userId`` 和
            ``itemId`` 列，其中 ``itemId`` 为以 ``;`` 分隔的多个物品 ID。

    Returns:
        dict[str, set[str]]: 用户到物品集合的映射。
    """
    df = pd.read_csv(filepath, usecols=["userId", "itemId"])
    # 分隔后展展，使用 groupby-集合 直接生成映射
    exploded = (
        df.assign(itemId=df["itemId"].str.split(";"))
        .explode("itemId")
        .dropna(subset=["itemId"])
    )
    exploded["itemId"] = exploded["itemId"].str.strip()
    mapping = (
        exploded.loc[exploded["itemId"] != ""]
        .groupby("userId")["itemId"]
        .apply(set)
        .to_dict()
    )
    return mapping


def preprocess_data(pairs):
    """
    将用户-物品对列表转换为用户到物品集合的映射。

    Args:
        pairs (list[tuple[str, str]]): 原始的用户-物品对列表。

    Returns:
        dict[str, set[str]]: 用户到物品集合的映射字典。
    """
    mapping = defaultdict(set)
    for uid, iid in pairs:
        mapping[uid].add(iid)
    return dict(mapping)


class UserCF:
    """
    基于用户的协同过滤推荐算法实现。

    Attributes:
        train_data (dict[str, set[str]]): 用户到物品集合的映射。
        similarity (str): 相似度计算方法，可选 'cosine' 或 'iif'。
        user_sim_matrix (dict[str, dict[str, float]]): 用户相似度矩阵，键为用户对 (u, v)，值为相似度分值。
    """
    def __init__(self, train_data, similarity='cosine'):
        self.train_data = train_data
        self.similarity = similarity
        self.user_sim_matrix = {}

    def _build_similarity(self):
        """
        构建用户相似度矩阵：
          1. 生成物品到用户的倒排索引；
          2. 计算用户共现矩阵（共现次数或 IIF 方式加权）；
          3. 对共现计数进行余弦归一化，得到用户相似度。
        """
        # 倒排索引：item_id -> set(user_id)
        item_user = defaultdict(set)
        for uid, items in self.train_data.items():
            for iid in items:
                item_user[iid].add(uid)

        # 累计用户共现次数
        co_counts = defaultdict(lambda: defaultdict(float))
        for users in item_user.values():
            user_list = list(users)
            weight = (
                1.0
                if self.similarity == "cosine"
                else 1.0 / math.log(1 + len(user_list))
            )
            for i, u in enumerate(user_list):
                u_dict = co_counts[u]
                for v in user_list[i + 1 :]:
                    u_dict[v] += weight
                    co_counts[v][u] += weight

        # 余弦归一化
        item_count = {u: len(items) for u, items in self.train_data.items()}
        for u, related in co_counts.items():
            nu = item_count.get(u, 0)
            if nu == 0:
                continue
            self.user_sim_matrix[u] = {}
            for v, cuv in related.items():
                nv = item_count.get(v, 0)
                self.user_sim_matrix[u][v] = (
                    cuv / math.sqrt(nu * nv) if nv > 0 else 0.0
                )


    def train(self):
        """
        计算并存储用户相似度矩阵，必须在调用推荐方法前执行。
        """
        self._build_similarity()

    def recommend(self, user, N=5, K=10):
        """
        为指定用户推荐物品。

        Args:
            user (str): 目标用户 ID。
            N (int): 返回的推荐物品数量。
            K (int): 参与计算的最相似用户数量。

        Returns:
            dict[str, float]: 推荐物品及其累加相似度得分，按得分降序排序后截取前 N 项。
        """
        rank = defaultdict(float)
        interacted = self.train_data.get(user, set())
        # 选取 K 个最相似用户
        neighbors = heapq.nlargest(
            K,
            self.user_sim_matrix.get(user, {}).items(),
            key=itemgetter(1),
        )

        # 汇总推荐分数
        for v, sim in neighbors:
            for iid in self.train_data.get(v, []):
                if iid in interacted:
                    continue
                rank[iid] += sim

        # 返回前 N 个物品
        return dict(heapq.nlargest(N, rank.items(), key=itemgetter(1)))

    def recommend_users(self, user, N=5):
        """
        为指定用户推荐相似用户。

        Args:
            user (str): 目标用户 ID。
            N (int): 返回的相似用户数量。

        Returns:
            list[str]: 按相似度降序排序的用户列表，长度不超过 N。
        """
        sims = self.user_sim_matrix.get(user, {})
        return [uid for uid, _ in heapq.nlargest(N, sims.items(), key=itemgetter(1))]


def save_similar_users(filepath, model, users, N=5):
    """
    将推荐的相似用户列表保存至 CSV 文件。

    Args:
        filepath (str): 输出 CSV 文件路径。
        model (UserCF): 已训练的 UserCF 模型实例。
        users (list[str]): 待处理的用户 ID 列表。
        N (int): 为每个用户推荐的相似用户数量。
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, mode='w', newline='', encoding='utf-8') as fw:
        writer = csv.writer(fw)
        writer.writerow(['userId', 'similarUsers'])
        for uid in users:
            similar = model.recommend_users(uid, N)
            writer.writerow([uid, ';'.join(map(str, similar))])
