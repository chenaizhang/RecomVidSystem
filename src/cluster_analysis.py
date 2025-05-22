import os
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


class ClusteringPipeline:
    """
    A pipeline for reading data, engineering features, performing MiniBatch-KMeans clustering, and saving results.
    """
    def __init__(self,
                 user_input_path: str,
                 item_input_path: str,
                 output_dir: str = 'output',
                 n_user_clusters: int = 4,
                 n_item_clusters: int = 50,
                 user_batch_size: int = 1024,
                 item_batch_size: int = 4096,
                 random_state: int = 42):
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

    def load_data(self):
        """Load user and item CSV files into pandas DataFrames."""
        self.users = pd.read_csv(self.user_input_path)
        self.items = pd.read_csv(self.item_input_path)

    @staticmethod
    def _count_splits(series: pd.Series) -> pd.Series:
        """Count the number of non-empty entries in a semicolon-separated string series."""
        return series.astype(str).apply(
            lambda s: len([x for x in s.split(';') if x.strip() != ''])
        )

    def engineer_user_features(self):
        """Engineer features for users: gender code, watch count, follow count, etc."""
        df = self.users.copy()
        df['gender_code'] = df['gender'].map({'M': 0, 'F': 1})
        df['num_watched'] = self._count_splits(df['itemId'])
        df['num_follow'] = self._count_splits(df['followId'])
        self.user_features = df[['age', 'fans', 'gender_code', 'num_watched', 'num_follow']]

    def engineer_item_features(self):
        """Engineer features for items: length, comment, like, watch, share, and viewer count."""
        df = self.items.copy()
        df['num_viewers'] = self._count_splits(df['userId'])
        self.item_features = df[['length', 'comment', 'like', 'watch', 'share', 'num_viewers']]

    def cluster_users(self):
        """Cluster users using MiniBatchKMeans and assign cluster labels."""
        self.user_km = MiniBatchKMeans(
            n_clusters=self.n_user_clusters,
            batch_size=self.user_batch_size,
            random_state=self.random_state
        )
        labels = self.user_km.fit_predict(self.user_features)
        self.users['cluster'] = labels

    def cluster_items(self):
        """Cluster items using MiniBatchKMeans and assign cluster labels."""
        self.item_km = MiniBatchKMeans(
            n_clusters=self.n_item_clusters,
            batch_size=self.item_batch_size,
            random_state=self.random_state
        )
        labels = self.item_km.fit_predict(self.item_features)
        self.items['cluster'] = labels

    def save_results(self):
        """Save the clustered users and items to CSV files in the output directory."""
        os.makedirs(self.output_dir, exist_ok=True)
        user_output = os.path.join(self.output_dir, 'F7.csv')
        item_output = os.path.join(self.output_dir, 'F6.csv')
        self.users.to_csv(user_output, index=False, encoding='utf-8')
        self.items.to_csv(item_output, index=False, encoding='utf-8')
        print(f"Saved user clusters to: {user_output}")
        print(f"Saved item clusters to: {item_output}")

    def run(self):
        """Execute the full pipeline: load data, engineer features, cluster, and save results."""
        self.load_data()
        self.engineer_user_features()
        self.engineer_item_features()
        self.cluster_users()
        self.cluster_items()
        self.save_results()

