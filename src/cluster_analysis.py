# src/cluster_analysis.py

import os
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


class ClusteringPipeline:
    """
    ClusteringPipeline：基于 MiniBatch-KMeans 的完整聚类流程，包括数据加载、特征工程、聚类计算与结果保存。

    属性:
        user_input_path (str)：用户数据 CSV 文件路径。
        item_input_path (str)：物品数据 CSV 文件路径。
        output_dir (str)：结果输出目录，默认 'output'。
        n_user_clusters (int)：用户聚类簇数，默认 4。
        n_item_clusters (int)：物品聚类簇数，默认 50。
        user_batch_size (int)：用户聚类的批处理大小，默认 1024。
        item_batch_size (int)：物品聚类的批处理大小，默认 4096。
        random_state (int)：随机种子，保证结果可复现，默认 42。

        users (pd.DataFrame)：原始用户数据 DataFrame。
        items (pd.DataFrame)：原始物品数据 DataFrame。
        user_features (pd.DataFrame)：用户特征矩阵。
        item_features (pd.DataFrame)：物品特征矩阵。
        user_km (MiniBatchKMeans)：用户聚类模型对象。
        item_km (MiniBatchKMeans)：物品聚类模型对象。
    """

    def __init__(
        self,
        user_input_path: str,
        item_input_path: str,
        output_dir: str = 'output',
        n_user_clusters: int = 4,
        n_item_clusters: int = 50,
        user_batch_size: int = 1024,
        item_batch_size: int = 4096,
        random_state: int = 42
    ):
        """
        初始化 ClusteringPipeline 实例。

        参数:
            user_input_path (str)：用户 CSV 文件路径，需包含 userId, itemId, followId 等字段。
            item_input_path (str)：物品 CSV 文件路径，需包含 length, comment, like, watch, share, userId 等字段。
            output_dir (str)：输出结果目录。
            n_user_clusters (int)：用户聚类簇数。
            n_item_clusters (int)：物品聚类簇数。
            user_batch_size (int)：用户聚类批处理大小。
            item_batch_size (int)：物品聚类批处理大小。
            random_state (int)：随机种子。
        """
        self.user_input_path = user_input_path
        self.item_input_path = item_input_path
        self.output_dir = output_dir
        self.n_user_clusters = n_user_clusters
        self.n_item_clusters = n_item_clusters
        self.user_batch_size = user_batch_size
        self.item_batch_size = item_batch_size
        self.random_state = random_state

        self.users = None
        self.items = None
        self.user_features = None
        self.item_features = None
        self.user_km = None
        self.item_km = None

    def load_data(self) -> None:
        """
        读取用户与物品的 CSV 数据文件，加载至 pandas DataFrame。

        加载后将数据分别存储在 self.users 和 self.items。
        """
        self.users = pd.read_csv(self.user_input_path)
        self.items = pd.read_csv(self.item_input_path)

    @staticmethod
    def _count_splits(series: pd.Series) -> pd.Series:
        """
        统计分号分隔字符串中的非空元素个数。

        参数:
            series (pd.Series)：包含分号分隔条目的字符串序列。

        返回:
            pd.Series：对应的元素计数。
        """
        return series.astype(str).apply(
            lambda s: len([x for x in s.split(';') if x.strip() != ''])
        )

    def engineer_user_features(self) -> None:
        """
        构建用户特征：年龄、粉丝数、性别编码、观看视频数与关注数。

        性别编码规则：M -> 0, F -> 1。
        num_watched 与 num_follow 分别统计 itemId 与 followId 列的分号分隔计数。
        """
        df = self.users.copy()
        # 性别映射
        df['gender_code'] = df['gender'].map({'M': 0, 'F': 1})
        # 统计观看视频数
        df['num_watched'] = self._count_splits(df['itemId'])
        # 统计关注数
        df['num_follow'] = self._count_splits(df['followId'])
        # 筛选特征列
        self.user_features = df[['age', 'fans', 'gender_code', 'num_watched', 'num_follow']]

    def engineer_item_features(self) -> None:
        """
        构建物品特征：视频时长、评论数、点赞数、观看数、分享数，以及观看该视频的用户数。

        num_viewers 通过统计 userId 列的分号分隔计数得到。
        """
        df = self.items.copy()
        # 统计观看该视频的不同用户数
        df['num_viewers'] = self._count_splits(df['userId'])
        # 筛选特征列
        self.item_features = df[['length', 'comment', 'like', 'watch', 'share', 'num_viewers']]

    def cluster_users(self) -> None:
        """
        对用户特征进行 MiniBatch-KMeans 聚类，并将聚类标签写入 self.users['cluster']。"""
        self.user_km = MiniBatchKMeans(
            n_clusters=self.n_user_clusters,
            batch_size=self.user_batch_size,
            random_state=self.random_state
        )
        labels = self.user_km.fit_predict(self.user_features)
        self.users['cluster'] = labels

    def cluster_items(self) -> None:
        """
        对物品特征进行 MiniBatch-KMeans 聚类，并将聚类标签写入 self.items['cluster']。"""
        self.item_km = MiniBatchKMeans(
            n_clusters=self.n_item_clusters,
            batch_size=self.item_batch_size,
            random_state=self.random_state
        )
        labels = self.item_km.fit_predict(self.item_features)
        self.items['cluster'] = labels

    def save_results(self) -> None:
        """
        将聚类后的用户与物品数据保存为 CSV 文件。

        输出路径:
          - 用户聚类结果: {output_dir}/F7.csv
          - 物品聚类结果: {output_dir}/F6.csv
        """
        os.makedirs(self.output_dir, exist_ok=True)
        user_csv = os.path.join(self.output_dir, 'F7.csv')
        item_csv = os.path.join(self.output_dir, 'F6.csv')
        # 保存文件，不包含索引
        self.users.to_csv(user_csv, index=False, encoding='utf-8')
        self.items.to_csv(item_csv, index=False, encoding='utf-8')
        print(f"用户聚类结果已保存至: {user_csv}")
        print(f"物品聚类结果已保存至: {item_csv}")

    def run(self) -> None:
        """
        执行完整聚类流程：加载数据 -> 特征工程 -> 聚类计算 -> 保存结果。
        """
        self.load_data()
        self.engineer_user_features()
        self.engineer_item_features()
        self.cluster_users()
        self.cluster_items()
        self.save_results()
