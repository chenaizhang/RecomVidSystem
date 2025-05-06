# RecomVidSystem
A system for personalized video recommendations, user behavior analysis, and predictive video popularity based on clustering techniques.

## Referance

UserCF：https://www.jianshu.com/p/7c5d9c008be9

## UserCF接口文档

#### 1. **`LoadDataFromCSV(filepath)`**

- **描述**: 从CSV文件中加载数据，数据格式为 `user, video_ids`，将视频ID进行拆分并返回处理后的数据集。

- **参数**:

  - `filepath` (str): CSV文件的路径，文件内容应该包含两列，分别为`user`（用户）和`video_ids`（视频ID）。

- **返回值**: 返回一个预处理过的数据集，格式为用户-视频的映射字典。

- **示例**:

  ```python
  train_data = LoadDataFromCSV("Data_backup.csv")
  ```

#### 2. **`PreProcessData(originData)`**

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

#### 3. **`UserCF` 类**

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

这个文档提供了完整的 `UserCF` 类的接口及其用法。