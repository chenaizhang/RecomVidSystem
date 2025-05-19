"""
Cluster Analysis & Visualization for Short‑Video Recommendation
==============================================================

*Updated: 2025‑05‑19*

This script now **integrates cluster visualization** directly, so running one
command will:

1. Load and clean `Data.csv`  ➜ `load_data`
2. Find/assign video clusters  ➜ `cluster_videos`
3. Build user–video matrix     ➜ `build_user_matrix`
4. Find/assign user clusters   ➜ `cluster_users`
5. **Plot** 2‑D projections of both video & user clusters  ➜ `plot_*`
6. Persist everything under `cluster_output/`

```bash
pip install pandas scikit-learn matplotlib umap-learn
python cluster_analysis.py --path Data.csv --plot
```

Add `--video_k` / `--user_k` to override *k*; omit to let the script select.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Iterable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 0.  Constants
# ---------------------------------------------------------------------------
DEFAULT_RANDOM_STATE = 42
FEAT_COLS = ["length", "comment", "like", "watch", "share"]

# ---------------------------------------------------------------------------
# 1.  Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_data(path: str | pathlib.Path) -> pd.DataFrame:
    """Load *Data.csv* and return cleaned DataFrame."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    # numeric coercion
    df[FEAT_COLS] = df[FEAT_COLS].apply(pd.to_numeric, errors="coerce")

    # parse semicolon‑separated user IDs → List[int]
    df["user_list"] = df["user_list"].apply(
        lambda x: [] if pd.isna(x) else [int(tok.strip()) for tok in str(x).split(";") if tok.strip().isdigit()]
    )

    return df

# ---------------------------------------------------------------------------
# 2.  Video clustering
# ---------------------------------------------------------------------------

def cluster_videos(
    df: pd.DataFrame,
    k: int,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEAT_COLS].fillna(0.0))
    model = KMeans(n_clusters=k, n_init="auto", random_state=random_state).fit(X)
    df["video_cluster"] = model.labels_
    return df, model, scaler

# ---------------------------------------------------------------------------
# 3.  User–video matrix & user clustering
# ---------------------------------------------------------------------------

def build_user_matrix(df: pd.DataFrame) -> pd.DataFrame:
    pairs: list[tuple[int, int]] = [
        (uid, vid)
        for vid, users in zip(df.id, df.user_list)
        for uid in users
    ]
    uv = pd.DataFrame(pairs, columns=["user_id", "video_id"])
    uv["watch"] = 1
    return uv.pivot_table(index="user_id", columns="video_id", values="watch", fill_value=0, aggfunc="max").astype(np.int8)


def cluster_users(
    user_video: pd.DataFrame,
    k: int,
    n_components: int | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    max_comp = min(50, user_video.shape[1] - 1)
    n_components = max_comp if n_components is None else min(n_components, max_comp)

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_red = svd.fit_transform(user_video.values)

    model = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labels = model.fit_predict(X_red)

    clusters = pd.DataFrame({"user_id": user_video.index, "user_cluster": labels})
    return clusters, model, svd

# ---------------------------------------------------------------------------
# 4.  Utilities for k selection
# ---------------------------------------------------------------------------

def find_best_k(X: np.ndarray, k_range: Iterable[int] = range(2, 11)) -> int:
    best_k, best_score = 0, -1
    for k in k_range:
        lab = KMeans(n_clusters=k, n_init="auto", random_state=DEFAULT_RANDOM_STATE).fit_predict(X)
        score = silhouette_score(X, lab)
        print(f"k={k:2d}  silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score
    return best_k

# ---------------------------------------------------------------------------
# 5.  Plotting helpers
# ---------------------------------------------------------------------------

def _scatter(
    coords: np.ndarray,
    labels: List[int] | np.ndarray,
    sizes: np.ndarray | None,
    title: str,
    out_path: pathlib.Path,
):
    plt.figure(figsize=(8, 6), dpi=120)
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=sizes, alpha=0.7, edgecolor="k")
    plt.title(title)
    plt.xlabel("Dim‑1")
    plt.ylabel("Dim‑2")
    plt.colorbar(sc, label="Cluster")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_video_clusters(
    df: pd.DataFrame,
    out_dir: pathlib.Path,
    method: str = "pca",
    random_state: int = DEFAULT_RANDOM_STATE,
):
    X = df[FEAT_COLS].fillna(0.0).values
    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=random_state)
    else:
        raise ValueError("method should be 'pca' or 'tsne'")
    coords = reducer.fit_transform(X)
    sizes = df["length"].fillna(0) * 2  # scale size by length
    _scatter(coords, df["video_cluster"], sizes, "Video clusters", out_dir / "video_clusters.png")


def plot_user_clusters(
    user_video: pd.DataFrame,
    user_clusters: pd.DataFrame,
    out_dir: pathlib.Path,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    svd = TruncatedSVD(n_components=min(50, user_video.shape[1] - 1), random_state=random_state)
    X_50 = svd.fit_transform(user_video.values)
    coords = TSNE(n_components=2, perplexity=50, random_state=random_state).fit_transform(X_50)
    labels = user_clusters.set_index("user_id").loc[user_video.index, "user_cluster"].values
    _scatter(coords, labels, None, "User clusters", out_dir / "user_clusters.png")

# ---------------------------------------------------------------------------
# 6.  CLI entry‑point
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Cluster videos & users and plot results")
    parser.add_argument("--path", default="Data.csv", help="CSV with raw data")
    parser.add_argument("--video_k", type=int, help="Number of video clusters")
    parser.add_argument("--user_k", type=int, help="Number of user clusters")
    parser.add_argument("--plot", action="store_true", help="Generate PNG plots")
    parser.add_argument("--out_dir", default="cluster_output", help="Output directory")
    args = parser.parse_args(argv)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load ------------------------------------------------------------------
    df = load_data(args.path)

    # 2) Video clusters --------------------------------------------------------
    X_video = df[FEAT_COLS].fillna(0.0).values
    if args.video_k is None:
        print("Selecting k for videos …")
        args.video_k = find_best_k(X_video)
        print(f"Optimal k_video = {args.video_k}\n")
    df, _, _ = cluster_videos(df, k=args.video_k)
    df.to_csv(out_dir / "videos_with_clusters.csv", index=False)

    # 3) User clusters ---------------------------------------------------------
    user_video = build_user_matrix(df)
    user_video.to_pickle(out_dir / "user_video.pkl")

    if args.user_k is None:
        print("Selecting k for users …")
        X_tmp = TruncatedSVD(n_components=min(50, user_video.shape[1]-1), random_state=DEFAULT_RANDOM_STATE).fit_transform(user_video.values)
        args.user_k = find_best_k(X_tmp)
        print(f"Optimal k_user = {args.user_k}\n")

    user_clusters, _, _ = cluster_users(user_video, k=args.user_k)
    user_clusters.to_csv(out_dir / "users_with_clusters.csv", index=False)

    # 4) Plotting --------------------------------------------------------------
    if args.plot:
        print("Creating plots …")
        plot_video_clusters(df, out_dir)
        plot_user_clusters(user_video, user_clusters, out_dir)
        print("Plots saved to", out_dir)

    print("\n🎉 Finished! All outputs in", out_dir.absolute())


if __name__ == "__main__":
    main()
