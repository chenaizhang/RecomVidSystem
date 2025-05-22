# src/itemcf_wals.py

import os
import csv
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


class WALSRecommender:
    """
    WALSRecommender：基于隐式反馈的加权交替最小二乘算法实现。

    属性:
        num_factors (int)：潜在因子维度。
        regularization (float)：L2 正则化系数。
        unobserved_weight (float)：对未观察到的交互的置信度权重。
        user_encoder (LabelEncoder)：用户ID编码器，将原始ID映射为整数索引。
        item_encoder (LabelEncoder)：物品ID编码器，将原始ID映射为整数索引。
        user_factors (np.ndarray)：训练后用户因子矩阵，形状 (n_users, num_factors)。
        item_factors (np.ndarray)：训练后物品因子矩阵，形状 (n_items, num_factors)。
        user_bias (np.ndarray)：训练后用户偏置向量，长度 n_users。
        item_bias (np.ndarray)：训练后物品偏置向量，长度 n_items。
        data (pd.DataFrame)：预处理后包含单条用户-物品交互行的数据框。
    """
    def __init__(
        self,
        num_factors: int = 32,
        regularization: float = 0.01,
        unobserved_weight: float = 0.1
    ):
        """
        初始化 WALSRecommender 模型及超参数。

        参数:
            num_factors (int)：潜在因子维度。
            regularization (float)：L2 正则化系数。
            unobserved_weight (float)：未观察交互的权重。
        """
        self.num_factors = num_factors
        self.regularization = regularization
        self.unobserved_weight = unobserved_weight
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.data = None

    def prepare_data(self, data: pd.DataFrame) -> tf.sparse.SparseTensor:
        """
        将原始 DataFrame 转换为 TensorFlow SparseTensor，用于模型训练。

        步骤:
        1. 对 itemId 列按分号拆分并展开为多行；
        2. 对用户和物品 ID 进行编码；
        3. 构造稀疏交互矩阵 (n_users, n_items)。

        参数:
            data (pd.DataFrame)：包含 ['userId','itemId']，itemId 为分号分隔字符串。

        返回:
            tf.sparse.SparseTensor：排序后的稀疏交互张量。
        """
        # 拆分 itemId 列并展开
        exploded = (
            data
            .assign(itemId=data['itemId'].str.split(';'))
            .explode('itemId')
            .reset_index(drop=True)
        )
        # 去除空白字符
        exploded['itemId'] = exploded['itemId'].str.strip()
        self.data = exploded

        # 编码用户和物品 ID
        user_indices = self.user_encoder.fit_transform(exploded['userId'])
        item_indices = self.item_encoder.fit_transform(exploded['itemId'])

        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)

        # 构建稀疏矩阵索引和值
        coords = np.column_stack((user_indices, item_indices))
        values = np.ones_like(user_indices, dtype=np.float32)
        shape = (n_users, n_items)

        sparse_tensor = tf.sparse.SparseTensor(coords, values, shape)
        return tf.sparse.reorder(sparse_tensor)

    def train(
        self,
        data: pd.DataFrame,
        num_iterations: int = 10,
        learning_rate: float = 0.01
    ) -> None:
        """
        使用 TensorFlow 优化器训练 WALS 模型。

        参数:
            data (pd.DataFrame)：原始交互数据。
            num_iterations (int)：迭代轮数。
            learning_rate (float)：Adam 优化器学习率。
        """
        sparse_tensor = self.prepare_data(data)
        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)

        # 初始化因子和偏置
        user_factors = tf.Variable(
            tf.random.normal([n_users, self.num_factors], dtype=tf.float32)
            * tf.sqrt(2.0 / (n_users + self.num_factors))
        )
        item_factors = tf.Variable(
            tf.random.normal([n_items, self.num_factors], dtype=tf.float32)
            * tf.sqrt(2.0 / (n_items + self.num_factors))
        )
        user_bias = tf.Variable(tf.zeros([n_users], dtype=tf.float32))
        item_bias = tf.Variable(tf.zeros([n_items], dtype=tf.float32))

        optimizer = tf.keras.optimizers.Adam(learning_rate)

        # 训练循环
        for epoch in range(1, num_iterations + 1):
            with tf.GradientTape() as tape:
                # 计算预测矩阵
                interaction = tf.matmul(user_factors, item_factors, transpose_b=True)
                preds = (
                    interaction
                    + tf.expand_dims(user_bias, 1)
                    + tf.expand_dims(item_bias, 0)
                )
                # 计算损失
                loss = self._compute_loss(
                    preds,
                    sparse_tensor,
                    user_factors,
                    item_factors,
                    user_bias,
                    item_bias
                )
            # 反向传播更新参数
            grads = tape.gradient(loss, [user_factors, item_factors, user_bias, item_bias])
            optimizer.apply_gradients(
                zip(grads, [user_factors, item_factors, user_bias, item_bias])
            )

            # 每 5 轮打印一次损失
            if epoch % 5 == 0:
                print(f"第 {epoch} 轮/{num_iterations}，损失: {loss:.4f}")

        # 保存训练结果
        self.user_factors = user_factors.numpy()
        self.item_factors = item_factors.numpy()
        self.user_bias = user_bias.numpy()
        self.item_bias = item_bias.numpy()

    def _compute_loss(
        self,
        preds: tf.Tensor,
        sparse: tf.sparse.SparseTensor,
        uf: tf.Variable,
        if_: tf.Variable,
        ub: tf.Variable,
        ib: tf.Variable
    ) -> tf.Tensor:
        """
        计算均方误差与 L2 正则化损失之和。

        参数:
            preds (tf.Tensor)：预测交互矩阵。
            sparse (tf.sparse.SparseTensor)：真实交互稀疏张量。
            uf, if_, ub, ib (tf.Variable)：因子矩阵与偏置向量。

        返回:
            tf.Tensor：标量损失。
        """
        dense_truth = tf.cast(tf.sparse.to_dense(sparse), tf.float32)
        # 均方误差
        mse_loss = tf.reduce_mean(tf.square(preds - dense_truth))
        # L2 正则化
        reg_loss = self.regularization * (
            tf.nn.l2_loss(uf) +
            tf.nn.l2_loss(if_) +
            tf.nn.l2_loss(ub) +
            tf.nn.l2_loss(ib)
        )
        return mse_loss + reg_loss

    def recommend_for_user(
        self,
        user_id: str,
        top_n: int = 5
    ) -> list[tuple[str, float]]:
        """
        为单个用户生成 Top-N 推荐列表。

        参数:
            user_id (str)：原始用户 ID。
            top_n (int)：推荐物品数量。

        返回:
            list[tuple[str, float]]：(物品ID, 归一化得分) 列表。
        """
        # 不存在的用户直接返回空
        if user_id not in self.user_encoder.classes_:
            return []

        uidx = self.user_encoder.transform([user_id])[0]
        # 计算得分：项乘用户因子 + 偏置
        raw_scores = (
            self.item_factors @ self.user_factors[uidx]
            + self.user_bias[uidx]
            + self.item_bias
        )

        # 排除已交互物品
        watched = set(self.data[self.data['userId'] == user_id]['itemId'])
        watched_idx = [
            self.item_encoder.transform([itm])[0]
            for itm in watched if itm in self.item_encoder.classes_
        ]
        if watched_idx:
            raw_scores[watched_idx] = -np.inf

        # 对有限值归一化至 [0,1]
        mask = np.isfinite(raw_scores)
        if mask.any():
            mn, mx = raw_scores[mask].min(), raw_scores[mask].max()
            denom = mx - mn if mx != mn else 1.0
            raw_scores[mask] = (raw_scores[mask] - mn) / denom

        # 部分排序选 Top-N
        k = min(top_n, raw_scores.size)
        top_idxs = np.argpartition(raw_scores, -k)[-k:]
        sorted_top = top_idxs[np.argsort(raw_scores[top_idxs])[::-1]]

        # 解码并返回
        return [
            (self.item_encoder.inverse_transform([idx])[0], float(raw_scores[idx]))
            for idx in sorted_top
        ]

    def recommend_all_and_save(
        self,
        top_n: int = 5,
        output_file: str = 'output/F4.csv'
    ) -> None:
        """
        为所有用户生成 Top-N 推荐并以长格式保存为 CSV：
        列格式 (userId, itemId, score)。

        参数:
            top_n (int)：每用户推荐数量。
            output_file (str): 输出文件路径。
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        start_time = time.time()

        # 计算完整评分矩阵
        scores = self.user_factors @ self.item_factors.T
        scores += self.user_bias[:, None] + self.item_bias[None, :]

        # 排除历史交互
        watch_map = self.data.groupby('userId')['itemId'].apply(list).to_dict()
        user_idx = {u: i for i, u in enumerate(self.user_encoder.classes_)}
        item_idx = {i: j for j, i in enumerate(self.item_encoder.classes_)}
        for uid, items in watch_map.items():
            u_i = user_idx[uid]
            mask_i = [item_idx[it] for it in items if it in item_idx]
            scores[u_i, mask_i] = -np.inf

        # 行归一化
        finite = np.isfinite(scores)
        mins = np.nanmin(np.where(finite, scores, np.nan), axis=1)
        maxs = np.nanmax(np.where(finite, scores, np.nan), axis=1)
        denom = np.where(maxs - mins == 0, 1.0, maxs - mins)
        scores = (scores - mins[:, None]) / denom[:, None]

        # 部分排序
        k = min(top_n, scores.shape[1])
        cand = np.argpartition(scores, -k, axis=1)[:, -k:]

        # 写入 CSV
        lines = ['userId,itemId,score']
        for u_i, user in enumerate(self.user_encoder.classes_):
            topc = cand[u_i]
            ordered = topc[np.argsort(scores[u_i, topc])[::-1]]
            for idx in ordered:
                it = self.item_encoder.inverse_transform([idx])[0]
                lines.append(f"{user},{it},{scores[u_i, idx]:.4f}")

        with open(output_file, 'w', newline='') as f:
            f.write('\n'.join(lines))

        print(f"推荐结果已保存至 {output_file}，耗时 {time.time()-start_time:.2f}s。")

    def save_retrieval(
        self,
        top_n: int = 10,
        output_file: str = 'output/Retrieval.csv'
    ) -> None:
        """
        将所有用户的 Top-N 推荐以宽表格式保存：
        userId,itemId1; itemId2; ...

        参数:
            top_n (int): 每用户推荐数量。
            output_file (str): 输出文件路径。
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        start_time = time.time()

        scores = self.user_factors @ self.item_factors.T
        scores += self.user_bias[:, None] + self.item_bias[None, :]
        watch_map = self.data.groupby('userId')['itemId'].apply(list).to_dict()
        user_idx = {u: i for i, u in enumerate(self.user_encoder.classes_)}
        item_idx = {i: j for j, i in enumerate(self.item_encoder.classes_)}
        for uid, items in watch_map.items():
            u_i = user_idx[uid]
            mask_i = [item_idx[it] for it in items if it in item_idx]
            scores[u_i, mask_i] = -np.inf

        finite = np.isfinite(scores)
        mins = np.nanmin(np.where(finite, scores, np.nan), axis=1)
        maxs = np.nanmax(np.where(finite, scores, np.nan), axis=1)
        denom = np.where(maxs - mins == 0, 1.0, maxs - mins)
        scores = (scores - mins[:, None]) / denom[:, None]

        k = min(top_n, scores.shape[1])
        cand = np.argpartition(scores, -k, axis=1)[:, -k:]

        lines = ['userId,itemId']
        for u_i, user in enumerate(self.user_encoder.classes_):
            topc = cand[u_i]
            ordered = topc[np.argsort(scores[u_i, topc])[::-1]]
            items = self.item_encoder.inverse_transform(ordered)
            lines.append(f"{user},{'; '.join(items)}")

        with open(output_file, 'w', newline='') as f:
            f.write('\n'.join(lines))

        print(f"宽表检索结果已保存至 {output_file}，共{len(self.user_encoder.classes_)}行，耗时 {time.time()-start_time:.2f}s。")
