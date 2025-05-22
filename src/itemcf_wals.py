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
    def __init__(self, num_factors=32, regularization=0.01, unobserved_weight=0.1):
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

    def prepare_data(self, data):
        data = data.assign(itemId=data['itemId'].str.split(';')) \
                   .explode('itemId') \
                   .reset_index(drop=True)
        data['itemId'] = data['itemId'].str.strip()
        self.data = data
        user_ids = self.user_encoder.fit_transform(data['userId'])
        item_ids = self.item_encoder.fit_transform(data['itemId'])
        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        indices = np.column_stack((user_ids, item_ids))
        values = np.ones(len(user_ids), dtype=np.float32)
        shape = (n_users, n_items)
        sparse = tf.sparse.SparseTensor(indices, values, shape)
        return tf.sparse.reorder(sparse)

    def train(self, data, num_iterations=10, learning_rate=0.01):
        sparse = self.prepare_data(data)
        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        user_factors = tf.Variable(
            tf.random.normal([n_users, self.num_factors], dtype=tf.float32) *
            tf.sqrt(2.0 / (n_users + self.num_factors)))
        item_factors = tf.Variable(
            tf.random.normal([n_items, self.num_factors], dtype=tf.float32) *
            tf.sqrt(2.0 / (n_items + self.num_factors)))
        user_bias = tf.Variable(tf.zeros([n_users], dtype=tf.float32))
        item_bias = tf.Variable(tf.zeros([n_items], dtype=tf.float32))
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        for itr in range(num_iterations):
            with tf.GradientTape() as tape:
                interaction = tf.matmul(user_factors, item_factors, transpose_b=True)
                preds = interaction + tf.expand_dims(user_bias,1) + tf.expand_dims(item_bias,0)
                loss = self._compute_loss(preds, sparse,
                                           user_factors, item_factors,
                                           user_bias, item_bias)
            grads = tape.gradient(loss, [user_factors, item_factors, user_bias, item_bias])
            optimizer.apply_gradients(zip(grads, [user_factors, item_factors, user_bias, item_bias]))
            # 每5次迭代打印一次进度和损失
            if (itr + 1) % 5 == 0:
                print(f"迭代 {itr + 1}/{num_iterations}, 损失: {loss:.4f}")
        self.user_factors = user_factors.numpy()
        self.item_factors = item_factors.numpy()
        self.user_bias = user_bias.numpy()
        self.item_bias = item_bias.numpy()

    def _compute_loss(self, preds, sparse,
                      uf, if_, ub, ib):
        preds = tf.cast(preds, tf.float32)
        dense = tf.cast(tf.sparse.to_dense(sparse), tf.float32)
        mse = tf.reduce_mean(tf.square(preds - dense))
        reg = self.regularization * (
            tf.reduce_sum(tf.square(uf)) +
            tf.reduce_sum(tf.square(if_)) +
            tf.reduce_sum(tf.square(ub)) +
            tf.reduce_sum(tf.square(ib)))
        return mse + reg

    def recommend_for_user(self, user_id, top_n=5):
        if user_id not in self.user_encoder.classes_:
            return []
        idx = self.user_encoder.transform([user_id])[0]
        raw = np.dot(self.item_factors, self.user_factors[idx]) + self.user_bias[idx] + self.item_bias
        watched = self.data[self.data['userId'] == user_id]['itemId'].unique()
        mask = [self.item_encoder.transform([i])[0] for i in watched if i in self.item_encoder.classes_]
        if mask:
            raw[np.array(mask)] = -np.inf
        finite = np.isfinite(raw)
        if finite.any():
            mn, mx = raw[finite].min(), raw[finite].max()
            raw[finite] = (raw[finite] - mn) / (mx - mn if mx != mn else 1)
        k = min(top_n, len(raw))
        part = np.argpartition(raw, -k)[-k:]
        idxs = part[np.argsort(raw[part])[::-1]]
        return list(zip(self.item_encoder.inverse_transform(idxs), raw[idxs]))

    def recommend_all_and_save(self, top_n=5, output_file='output/F4.csv'):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        start = time.time()
        scores = self.user_factors @ self.item_factors.T
        scores += self.user_bias[:,None] + self.item_bias[None,:]
        watch_map = self.data.groupby('userId')['itemId'].apply(list).to_dict()
        u_to_idx = {u: i for i, u in enumerate(self.user_encoder.classes_)}
        i_to_idx = {i: j for j, i in enumerate(self.item_encoder.classes_)}
        for u, items in watch_map.items():
            ui = u_to_idx[u]
            idxs = [i_to_idx[it] for it in items if it in i_to_idx]
            scores[ui, idxs] = -np.inf
        finite = np.isfinite(scores)
        mins = np.nanmin(np.where(finite, scores, np.nan), axis=1)
        maxs = np.nanmax(np.where(finite, scores, np.nan), axis=1)
        denom = np.where(maxs - mins == 0, 1, maxs - mins)
        scores = (scores - mins[:,None]) / denom[:,None]
        part = np.argpartition(scores, -top_n, axis=1)[:, -top_n:]
        lines = ['userId,itemId,score']
        for ui, u in enumerate(self.user_encoder.classes_):
            cand = part[ui]
            ordered = cand[np.argsort(scores[ui, cand])[::-1]]
            for it in ordered:
                lines.append(f"{u},{self.item_encoder.inverse_transform([it])[0]},{scores[ui,it]:.4f}")
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        print(f"推荐结果已保存至 {output_file}，耗时 {time.time()-start:.2f}s.")

    def save_retrieval(self, top_n=10, output_file='output/Retrieval.csv'):
        """
        将每个用户的前 top_n 个推荐以宽表形式保存：
        userId,itemId1; itemId2; ...
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        start = time.time()
        scores = self.user_factors @ self.item_factors.T
        scores += self.user_bias[:,None] + self.item_bias[None,:]
        watch_map = self.data.groupby('userId')['itemId'].apply(list).to_dict()
        u_to_idx = {u: i for i, u in enumerate(self.user_encoder.classes_)}
        i_to_idx = {i: j for j, i in enumerate(self.item_encoder.classes_)}
        for u, items in watch_map.items():
            ui = u_to_idx[u]
            idxs = [i_to_idx[it] for it in items if it in i_to_idx]
            scores[ui, idxs] = -np.inf
        finite = np.isfinite(scores)
        mins = np.nanmin(np.where(finite, scores, np.nan), axis=1)
        maxs = np.nanmax(np.where(finite, scores, np.nan), axis=1)
        denom = np.where(maxs - mins == 0, 1, maxs - mins)
        scores = (scores - mins[:,None]) / denom[:,None]
        k = min(top_n, scores.shape[1])
        part = np.argpartition(scores, -k, axis=1)[:, -k:]
        users = self.user_encoder.classes_
        lines = ['userId,itemId']
        for ui, u in enumerate(users):
            cand = part[ui]
            ordered = cand[np.argsort(scores[ui, cand])[::-1]]
            items = self.item_encoder.inverse_transform(ordered)
            lines.append(f"{u},{'; '.join(map(str, items))}")
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        print(f"宽表检索结果已保存至 {output_file}，共{len(users)}行，耗时 {time.time()-start:.2f}s.")


