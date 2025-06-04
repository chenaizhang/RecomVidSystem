"""Main entry and reusable pipeline functions."""

import os
import pandas as pd
import time
from usercf import load_data_from_csv, UserCF, save_similar_users
from itemcf_wals import WALSRecommender
from item_popularity import ItemPopularityCalculator
from cluster_analysis import ClusteringPipeline


def run_usercf() -> pd.DataFrame:
    """Run user-based collaborative filtering and save ``F3.csv``.

    Returns
    -------
    pd.DataFrame
        DataFrame of the generated ``F3.csv`` file.
    """
    train = load_data_from_csv("data/user.csv")
    model = UserCF(train)
    model.train()
    save_similar_users("output/F3.csv", model, list(train.keys()), N=10)
    return pd.read_csv("output/F3.csv")


def run_wals() -> pd.DataFrame:
    """Run WALS recommendation and save ``F4.csv``/``Retrieval.csv``."""
    data = pd.read_csv("data/user.csv")
    recommender = WALSRecommender(num_factors=32)
    recommender.train(data, num_iterations=20, learning_rate=0.001)
    recommender.recommend_all_and_save(top_n=5, output_file="output/F4.csv")
    recommender.save_retrieval(top_n=100, output_file="output/Retrieval.csv")
    return pd.read_csv("output/F4.csv")


def run_popularity() -> pd.DataFrame:
    """Compute video popularity prediction and save ``F5.csv``."""
    calculator = ItemPopularityCalculator(
        input_pattern="output/Retrieval.csv",
        output_file="output/F5.csv",
    )
    calculator.run()
    return pd.read_csv("output/F5.csv")


def run_clustering() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run user/item clustering and save ``F6.csv`` and ``F7.csv``."""
    pipeline = ClusteringPipeline(
        user_input_path="data/user.csv",
        item_input_path="data/item.csv",
        output_dir="output",
        n_user_clusters=20,
        n_item_clusters=50,
    )
    pipeline.run()
    user_csv = os.path.join("output", "F7.csv")
    item_csv = os.path.join("output", "F6.csv")
    return pd.read_csv(user_csv), pd.read_csv(item_csv)


def run_all():
    """Execute all steps sequentially and return generated DataFrames."""
    f3 = run_usercf()
    f4 = run_wals()
    f5 = run_popularity()
    f7, f6 = run_clustering()
    return f3, f4, f5, f7, f6

if __name__ == "__main__":
    run_all()
