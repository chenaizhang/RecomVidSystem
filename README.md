# RecomVidSystem

A system for personalized video recommendations, user behavior analysis, and predictive video popularity based on clustering techniques.

## Referance

UserCF: https://www.jianshu.com/p/7c5d9c008be9
ItemCF: https://www.jianshu.com/p/f306a37a7374

## UserCF接口文档

### 1. **`LoadDataFromCSV(filepath)`**

- **描述**: 从CSV文件中加载数据，数据格式为 `user, video_ids`，将视频ID进行拆分并返回处理后的数据集。

- **参数**:

  - `filepath` (str): CSV文件的路径，文件内容应包含两列，分别为 `user`（用户）和 `video_ids`（视频ID）。

- **返回值**: 返回一个预处理过的数据集，格式为用户-视频的映射字典。

- **示例**:

  ```python
  train_data = LoadDataFromCSV("Data_backup.csv")
  ```

### 2. **`PreProcessData(originData)`**

- **描述**: 将原始数据转换为用户-视频的映射字典，映射字典格式为：

  ```python
  {"User1": {VideoID1, VideoID2, VideoID3,...},
   "User2": {VideoID4, VideoID5, VideoID6,...}, ...}
  ```

- **参数**:

  - `originData` (list): 原始数据（用户与视频ID的对），每个元素为 `[user, item]`。

- **返回值**: 返回一个用户与视频集合的映射字典。

- **示例**:

  ```python
  train_data = PreProcessData(origin_data)
  ```

### 3. **`UserCF` 类**

**描述**: `UserCF` 是一个基于用户协同过滤的推荐算法类，支持基于用户相似度进行推荐。

- **`__init__(self, trainData, similarity="cosine")`**

  - **描述**: 初始化 `UserCF` 类，设置训练数据和相似度计算方式。

  - **参数**:

    - `trainData` (dict): 训练数据，用户-视频的映射字典。
    - `similarity` (str): 相似度计算方式，支持 `cosine`（余弦相似度）和 `iif`（倒数频率加权）。

  - **返回值**: 无返回值。

  - **示例**:

    ```python
    user_cf = UserCF(train_data)
    ```

- **`similarity(self)`**

  - **描述**: 计算用户之间的相似度矩阵，支持 `cosine` 或 `iif` 相似度计算方法。

  - **返回值**: 无返回值，更新内部的 `_userSimMatrix`（用户相似度矩阵）。

  - **示例**:

    ```python
    user_cf.similarity()
    ```

- **`recommend(self, user, N, K)`**

  - **描述**: 为指定用户生成视频推荐，基于用户相似度和其他用户的行为进行推荐。

  - **参数**:

    - `user` (str): 被推荐的用户ID。
    - `N` (int): 推荐的商品个数。
    - `K` (int): 查找的最相似用户个数。

  - **返回值**: 返回一个字典，包含推荐的物品和相应的兴趣度分数，按兴趣度排序。

  - **示例**:

    ```python
    recommendations = user_cf.recommend("User1", 5, 80)
    print(recommendations)
    ```

- **`recommendperson(self, user, N)`**

  - **描述**: 推荐与指定用户最相似的 N 个其他用户。

  - **参数**:

    - `user` (str): 被推荐的用户ID。
    - `N` (int): 推荐的相似用户个数。

  - **返回值**: 返回一个列表，包含与指定用户相似度最高的 N 个用户，按相似度从高到低排序。

  - **示例**:

    ```python
    recommended_users = user_cf.recommendperson("User1", 3)
    print(f"Recommended users similar to User1: {recommended_users}")
    ```

- **`train(self)`**

  - **描述**: 训练模型，计算用户之间的相似度。

  - **返回值**: 无返回值，更新用户相似度矩阵。

  - **示例**:

    ```python
    user_cf.train()
    ```

#### 4. **示例代码**

**加载数据并进行模型训练和推荐**:

```python
# 加载训练数据
train_data = LoadDataFromCSV("Data_backup.csv")
print(f"Train data size: {len(train_data)}")

# 初始化并训练UserCF模型
user_cf = UserCF(train_data)
user_cf.train()

# 推荐前5个用户的视频
for user in list(train_data.keys())[:5]:
    print(f"Recommendations for {user}: {user_cf.recommend(user, 5, 80)}")
```

#### 5. **依赖库**

- **pandas**: 用于加载和处理CSV数据。
- **collections**: 用于创建字典并初始化默认值。
- **operator**: 用于排序操作。
- **math**: 用于数学计算（如平方根、对数等）。
- **time**: 用于计时。

#### 6. **返回值和结果**

- `LoadDataFromCSV` 和 `PreProcessData` 返回的数据是以字典的形式组织的，用户作为键，视频ID集合作为值。
- `recommend` 返回一个字典，包含按兴趣度排序的推荐视频。
- `recommendperson` 返回一个包含与指定用户相似度最高的 N 个用户的列表。返回的列表是按相似度从高到低排序的。

## RecomVidSystem · Developer Documentation

一个基于 **Python + Apache Spark** 的短视频推荐实验框架，支持

- **UserCF**（用户协同过滤）
- **ItemCF‑ALS**（基于交替最小二乘的物品协同过滤）
- **批量离线召回脚本**
- 多值列（`a;b;c`）自动展开、字符串/数字 ID 混合
- 去历史过滤、批量输出带得分的 Top‑K 推荐

> ⚠️ 本文档承接前面「UserCF 接口文档」继续撰写，新增 **ItemCF‑ALS API、CLI、数据格式与运行指南**。

------

## 目录

1. [环境依赖](https://chatgpt.com/c/682a7fe0-116c-8002-ba44-f411e8f74e5a#环境依赖)
2. [数据格式约定](https://chatgpt.com/c/682a7fe0-116c-8002-ba44-f411e8f74e5a#数据格式约定)
3. [ItemCF‑ALS 接口文档](https://chatgpt.com/c/682a7fe0-116c-8002-ba44-f411e8f74e5a#itemcf‑als-接口文档)
4. [批量推荐脚本 `generate_recs.py`](https://chatgpt.com/c/682a7fe0-116c-8002-ba44-f411e8f74e5a#批量推荐脚本-generaterecspy)
5. [快速开始](https://chatgpt.com/c/682a7fe0-116c-8002-ba44-f411e8f74e5a#快速开始)
6. [评估指标](https://chatgpt.com/c/682a7fe0-116c-8002-ba44-f411e8f74e5a#评估指标)
7. [参考链接](https://chatgpt.com/c/682a7fe0-116c-8002-ba44-f411e8f74e5a#参考链接)

------

## 环境依赖

| 组件             | 版本建议                          | 用途                   |
| ---------------- | --------------------------------- | ---------------------- |
| **Python**       | 3.9 +                             | 运行脚本               |
| **Apache Spark** | 3.3 + / Stand‑alone 或 local 模式 | 分布式/本地计算        |
| **PySpark**      | 与 Spark 版本一致                 | Python API             |
| **pandas**       | ≥ 1.5                             | CSV 预处理（UserCF）   |
| **numpy**        | ≥ 1.22                            | 向量运算（ALS 相似度） |

```bash
pip install pyspark pandas numpy
```

------

## 数据格式约定

### 交互日志（长表）

```csv
userId,itemId,rating
U1,42,1
U1,99,1
U2,42,1
…
```

### 宽表（多值列）

```csv
userId,itemId
A,a;b;d
B,a;c;d
C,b;e
```

脚本自动按 `--multiDelim ';'` 拆分为长表；缺失 `rating` 列时默认填 1.0。

------

## ItemCF‑ALS 接口文档

封装于 **`itemcf_als.py`**，核心类 `ItemCF_ALS`。

### 1. `ItemCF_ALS.__init__(spark, rank=50, reg_param=0.01, max_iter=10, implicit_prefs=False)`

| 参数             | 说明                                                |
| ---------------- | --------------------------------------------------- |
| `spark`          | 已创建的 `SparkSession`                             |
| `rank`           | 隐因子维度 K                                        |
| `reg_param`      | 正则系数 λ                                          |
| `max_iter`       | ALS 迭代轮数                                        |
| `implicit_prefs` | True = Implicit ALS（信任度），False = Explicit ALS |

### 2. `train(ratings_df, user_col="userId", item_col="itemId", multi_delim=None)`

| 参数          | 说明                                                         |
| ------------- | ------------------------------------------------------------ |
| `ratings_df`  | `DataFrame`，包含 `userId`,`itemId`[,`rating`]               |
| `multi_delim` | 若 `item_col` 是分号分隔的多值列，传入 `';'` 自动 `explode()` |

**效果**：

1. `trim()` 去首尾空格；
2. 若无 `rating` 列 → 补 `1.0`；
3. `StringIndexer` 将原始 ID → 连续整数索引；
4. 训练 Spark ALS，缓存物品向量。

### 3. `get_similar_items(raw_item_id, top_k=10)`

返回与 `raw_item_id` 最相似的 **Top‑K** 物品及余弦分数：

```python
>>> rec.get_similar_items("a", 3)
[("d", 0.9996), ("e", 0.9933), ("c", 0.9920)]
```

### 4. `recommend_for_user(raw_user_id, num_items=10)`

基于 ALS 预测分数，为单个用户推荐 **num_items** 条新视频（已过滤历史）：

```python
>>> rec.recommend_for_user("A", 5)
[("c", 5.4431), ("e", 4.9920), …]
```

------

## 批量推荐脚本 `generate_recs.py`

> 适合离线产出「全量用户 × Top‑K 推荐」CSV。

### CLI 参数

| 参数                       | 默认         | 作用                 |
| -------------------------- | ------------ | -------------------- |
| `--ratings`                | *(必填)*     | 交互 CSV 路径        |
| `--multiDelim`             | *(空)*       | 多值列分隔符，如 `;` |
| `--topK`                   | 10           | 每用户保存 K 条推荐  |
| `--output`                 | `output.csv` | 结果目录前缀         |
| 其余 `--rank --reg --iter` | 同 ALS 超参  |                      |

### 输出

```csv
userId,itemId_score_list
A,"c:5.4431;e:4.9920;…"
B,"b:5.3874;e:4.9513;…"
```

- 分号分隔；`itemId:score` 形式；
- 已去除用户看过的视频；
- `score` 保留 4 位小数。

------

## 快速开始

```bash
# 1. 本地模式跑一个小例子
spark-submit itemcf_als.py                \
  --ratings 1.csv                         \
  --multiDelim ';'                        \
  --itemId a --topK 5

# 2. 产出全量用户 Top‑10 推荐
spark-submit generate_recs.py             \
  --ratings Data_backup.csv               \
  --multiDelim ';'                        \
  --topK 10                               \
  --output output.csv
```

------

## 评估指标

| 指标            | 公式                                          | 说明                        |
| --------------- | --------------------------------------------- | --------------------------- |
| **Precision@K** | TP / K                                        | 推荐列表前 K 中命中的比例   |
| **Recall@K**    | TP / GT                                       | 命中数 ÷ 用户实际喜欢的总数 |
| **MAP@K**       | ( \frac{1}{                                   | U                           |
| **NDCG@K**      | $\frac{1}{IDCG}\sum\frac{rel_i}{\log_2(i+1)}$ | 把正样本按位置折损          |

实现示例详见 `metrics.py`（TODO）。

------

## 参考链接

- [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
- Spark MLlib 官方文档：https://spark.apache.org/docs/latest/ml-collaborative-filtering.html
- Jiashu 相关文章：
  - UserCF https://www.jianshu.com/p/7c5d9c008be9
  - ItemCF https://www.jianshu.com/p/f306a37a7374