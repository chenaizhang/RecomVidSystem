import argparse
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

def load_item_features(path: str) -> pd.DataFrame:
    """
    从 item.csv 中读取数据并构造特征：
    - length, comment, like, watch, share
    - num_viewers: userId 列拆分计数
    """
    df = pd.read_csv(path)
    # 统计被多少用户看过
    df['num_viewers'] = (
        df['userId']
        .astype(str)
        .apply(lambda s: len([x for x in s.split(';') if x.strip() != '']))
    )
    return df[['length', 'comment', 'like', 'watch', 'share', 'num_viewers']]

def plot_elbow(features: pd.DataFrame, k_min: int, k_max: int):
    """
    对 k 从 k_min 到 k_max 计算 inertia 并绘图
    """
    ks = list(range(k_min, k_max + 1))
    inertias = []
    for k in ks:
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=42)
        km.fit(features)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, '-o')
    plt.xticks(ks)
    plt.xlabel('Number of clusters $k$')
    plt.ylabel('Inertia (SSE)')
    plt.title('Elbow Method on item.csv Features')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Elbow method for item.csv 聚类簇数选择')
    parser.add_argument('--input', type=str, default='data/item.csv',
                        help='item.csv 文件路径')
    parser.add_argument('--min_k', type=int, default=1,
                        help='最小簇数 (inclusive)')
    parser.add_argument('--max_k', type=int, default=100,
                        help='最大簇数 (inclusive)')
    args = parser.parse_args()

    feats = load_item_features(args.input)
    plot_elbow(feats, args.min_k, args.max_k)
