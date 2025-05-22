# RecomVidSystem

A system for personalized video recommendations, user behavior analysis, and predictive video popularity based on clustering techniques.

## Referance

UserCF: https://www.jianshu.com/p/7c5d9c008be9

## 短视频推荐系统组件说明

本项目由多个模块组成，依次完成加载数据、协同过滤推荐、WALS 推荐、视频热度统计、聚类分析等功能。以下为各模块的接口和使用示例。

---

### 安装与依赖

```bash
pip install pandas numpy scikit-learn tensorflow
```

确保以下文件在项目路径下：

* `data/Data_ui.csv`：用户-视频交互宽表，格式 `userId,itemId1;itemId2;...`
* `data/user.csv`：用户属性数据
* `data/item.csv`：视频属性数据

---

### 模块与接口

#### 1. `usercf.py`

##### 函数

* `load_data_from_csv(filepath: str) -> dict[str, set[str]]`

  * 从 CSV 加载宽表，将 `itemId` 列按 `;` 拆分，返回 `{userId: {itemId}}` 映射。

* `save_similar_users(filepath: str, model: UserCF, users: list[str], N: int) -> None`

  * 将为每个用户推荐的相似用户列表保存为 CSV，格式 `userId,similarUsers1;similarUsers2;...`。

##### 类

* `UserCF(train_data: dict[str, set[str]], similarity: str = 'cosine')`

  * 基于用户的协同过滤算法
  * `train() -> None`：构建用户相似度矩阵
  * `recommend(user: str, N: int = 5, K: int = 10) -> dict[str, float]`：推荐 Top-N 未交互物品
  * `recommend_users(user: str, N: int = 5) -> list[str]`：推荐 Top-N 相似用户

---

#### 2. `itemcf_wals.py`

##### 类

* `WALSRecommender(num_factors: int = 32, regularization: float = 0.01, unobserved_weight: float = 0.1)`

  * 加权交替最小二乘（WALS）模型实现
  * `train(data: pd.DataFrame, num_iterations: int = 10, learning_rate: float = 0.01) -> None`：训练模型
  * `recommend_for_user(user_id: str, top_n: int = 5) -> list[tuple[str, float]]`：单用户推荐 Top-N
  * `recommend_all_and_save(top_n: int, output_file: str) -> None`：为所有用户生成长格式推荐并保存 CSV
  * `save_retrieval(top_n: int, output_file: str) -> None`：为所有用户生成宽表形式 Top-N 列表并保存 CSV

---

#### 3. `item_popularity.py`

##### 类

* `ItemPopularityCalculator(input_pattern: str, output_file: str)`

  * 统计每个视频被不同用户推荐的次数（热度）
  * `load_data() -> self`：读取 glob 匹配的 CSV 文件
  * `explode_items() -> self`：拆分并展开 `itemId` 列
  * `compute_popularity() -> self`：按 `itemId` 分组，统计唯一 `userId` 数量
  * `save() -> self`：将热度统计结果保存为 CSV
  * `run() -> self`：一键执行完整流程

---

#### 4. `cluster_analysis.py`

##### 类

* `ClusteringPipeline(user_input_path: str,
                      item_input_path: str,
                      output_dir: str = 'output',
                      n_user_clusters: int = 4,
                      n_item_clusters: int = 50,
                      user_batch_size: int = 1024,
                      item_batch_size: int = 4096,
                      random_state: int = 42)`

  * MiniBatch-KMeans 聚类流程
  * `load_data() -> None`：读取用户与视频属性文件
  * `engineer_user_features() -> None`：构建用户特征矩阵
  * `engineer_item_features() -> None`：构建视频特征矩阵
  * `cluster_users() -> None`：对用户进行聚类并写入 `self.users['cluster']`
  * `cluster_items() -> None`：对视频进行聚类并写入 `self.items['cluster']`
  * `save_results() -> None`：保存聚类结果到 CSV (`F7.csv`, `F6.csv`)
  * `run() -> None`：执行完整聚类流程

---

### 主程序 `main.py` 示例

```python
import time
import pandas as pd
from usercf import load_data_from_csv, UserCF, save_similar_users
from itemcf_wals import WALSRecommender
from item_popularity import ItemPopularityCalculator
from cluster_analysis import ClusteringPipeline

if __name__ == '__main__':
    # F3：相似用户推荐
    train = load_data_from_csv('data/Data_ui.csv')
    model = UserCF(train)
    model.train()
    save_similar_users('output/F3.csv', model, list(train.keys()), N=10)

    # F4：WALS 视频推荐
    data = pd.read_csv('data/Data_ui.csv')
    wals = WALSRecommender(num_factors=32)
    wals.train(data, num_iterations=20, learning_rate=0.001)
    wals.recommend_all_and_save(5, 'output/F4.csv')
    wals.save_retrieval(100, 'output/Retrieval.csv')

    # F5：视频热度统计
    calc = ItemPopularityCalculator('output/Retrieval.csv', 'output/F5.csv')
    calc.run()

    # F6-F7：用户 & 视频聚类
    pipeline = ClusteringPipeline(
        user_input_path='data/user.csv',
        item_input_path='data/item.csv',
        output_dir='output',
        n_user_clusters=4,
        n_item_clusters=30
    )
    pipeline.run()
```

---

更多详细说明请参阅各模块源代码中的文档字符串。
