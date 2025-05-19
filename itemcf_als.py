# itemcf_als.py
"""
Item‑based Collaborative Filtering on Spark (String‑ID Friendly)
===============================================================

### v2 — Robust to whitespace & clearer errors
* **始终去空格**：不论有没有 `--multiDelim`，都会 `trim()` `userId`、`itemId`，避免因隐形空格导致 "ID not seen"。
* **容错 index_of()**：自动 `strip()` 调用参数；若仍找不到，打印可用 ID 示例方便排错。
* 其他接口、命令行保持不变。

---
```bash
spark-submit itemcf_als.py \
  --ratings 1.csv \
  --multiDelim ';' \
  --itemId a \
  --topK 10
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, explode, lit, split, trim

# ----------------------------------------------------------------------------
# Helper dataclasses ---------------------------------------------------------
# ----------------------------------------------------------------------------


class _IndexerBundle:
    """Hold mappings for raw⇄index look‑ups."""

    def __init__(self, id2raw: Dict[int, str], raw2id: Dict[str, int]):
        self.id2raw = id2raw
        self.raw2id = raw2id

    def index_of(self, raw: str) -> int:
        raw = raw.strip()
        if raw not in self.raw2id:
            _debug_missing(raw, self.raw2id)
        return self.raw2id[raw]

    def raw_of(self, idx: int) -> str:
        return self.id2raw[idx]


# ----------------------------------------------------------------------------
# Main recommender -----------------------------------------------------------
# ----------------------------------------------------------------------------


class ItemCF_ALS:
    """ALS‑based item‑item recommender that handles string IDs and multi‑value cols."""

    def __init__(
        self,
        spark: SparkSession,
        rank: int = 50,
        reg_param: float = 0.01,
        max_iter: int = 10,
        implicit_prefs: bool = False,
    ) -> None:
        self.spark = spark
        self.rank = rank
        self.reg_param = reg_param
        self.max_iter = max_iter
        self.implicit_prefs = implicit_prefs

        self.model: ALSModel | None = None
        self.user_map: _IndexerBundle | None = None
        self.item_map: _IndexerBundle | None = None
        self._item_vecs: Dict[int, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        ratings: DataFrame,
        user_col: str = "userId",
        item_col: str = "itemId",
        multi_delim: str | None = None,
    ) -> None:
        """Fit ALS on `ratings` (auto‑handle explode, whitespace, rating=1)."""

        df = ratings

        # 1) explode per‑row multiple items
        if multi_delim:
            df = df.withColumn(item_col, explode(split(col(item_col), multi_delim)))

        # 2) global trim to kill hidden spaces
        df = df.withColumn(item_col, trim(col(item_col))).withColumn(user_col, trim(col(user_col)))

        # 3) add rating=1.0 if missing
        if "rating" not in df.columns:
            df = df.withColumn("rating", lit(1.0))

        # 4) string indexer
        user_idxer = StringIndexer(inputCol=user_col, outputCol="userIdx", handleInvalid="error")
        item_idxer = StringIndexer(inputCol=item_col, outputCol="itemIdx", handleInvalid="error")
        df_idx = item_idxer.fit(df).transform(user_idxer.fit(df).transform(df))

        # 5) persist mappings
        self.user_map = _collect_mapping(df_idx, "userIdx", user_col)
        self.item_map = _collect_mapping(df_idx, "itemIdx", item_col)

        # 6) ALS
        als = ALS(
            userCol="userIdx",
            itemCol="itemIdx",
            ratingCol="rating",
            rank=self.rank,
            maxIter=self.max_iter,
            regParam=self.reg_param,
            implicitPrefs=self.implicit_prefs,
            coldStartStrategy="drop",
        )
        self.model = als.fit(df_idx.cache())
        self._cache_item_vectors()

    def get_similar_items(self, raw_item_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        _ensure(self._item_vecs, "Model not trained: call train() first.")
        idx = self.item_map.index_of(raw_item_id)
        vec = self._item_vecs[idx]
        vnorm = np.linalg.norm(vec)
        sims: List[Tuple[int, float]] = []
        for j, v in self._item_vecs.items():
            if j == idx:
                continue
            sims.append((j, float(np.dot(vec, v) / (vnorm * np.linalg.norm(v)))))
        sims.sort(key=lambda x: x[1], reverse=True)
        return [(self.item_map.raw_of(j), s) for j, s in sims[:top_k]]

    def recommend_for_user(self, raw_user_id: str, num_items: int = 10) -> List[Tuple[str, float]]:
        _ensure(self.model, "Model not trained: call train() first.")
        uidx = self.user_map.index_of(raw_user_id)
        recs_df = (
            self.model.recommendForUserSubset(
                self.spark.createDataFrame([(uidx,)], ["userIdx"]), num_items
            )
            .select("recommendations")
            .limit(1)
        )
        if recs_df.count() == 0:
            return []
        rows = recs_df.collect()[0][0]
        return [(self.item_map.raw_of(r[0]), r[1]) for r in rows]

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _cache_item_vectors(self) -> None:
        _ensure(self.model, "ALS model not fitted.")
        self._item_vecs = {
            int(r["id"]): np.array(r["features"], dtype=np.float32)
            for r in self.model.itemFactors.collect()
        }


# ----------------------------------------------------------------------------
# Utility fns -----------------------------------------------------------------
# ----------------------------------------------------------------------------

def _collect_mapping(df: DataFrame, idx_col: str, raw_col: str) -> _IndexerBundle:
    mapping = df.select(idx_col, raw_col).distinct().collect()
    id2raw = {int(r[idx_col]): r[raw_col] for r in mapping}
    raw2id = {v: k for k, v in id2raw.items()}
    return _IndexerBundle(id2raw, raw2id)


def _ensure(obj, msg="") -> None:
    if obj is None:
        raise RuntimeError(msg)


def _debug_missing(raw: str, mapping: Dict[str, int]):
    """Raise friendly error with closest matches."""
    import difflib, textwrap

    suggestions = difflib.get_close_matches(raw, mapping.keys(), n=5, cutoff=0.0)
    sample = list(mapping.keys())[:10]
    raise ValueError(
        textwrap.dedent(
            f"""
            ID '{raw}' not found in training data.\n
            ➤ 可能原因：\n   • CSV 中带隐藏空格；已全局 trim，但请再次确认。\n   • 训练文件与查询参数大小写不一致。\n
            ➤ 可用 itemId 样例： {sample} ...\n            ➤ 相近拼写建议： {suggestions}\n            """
        )
    )


# ----------------------------------------------------------------------------
# CLI driver & self‑test ------------------------------------------------------
# ----------------------------------------------------------------------------

def _parse_cli() -> argparse.Namespace | None:
    import sys
    if len(sys.argv) == 1:
        return None
    p = argparse.ArgumentParser("ALS ItemCF (string ID ready)")
    p.add_argument("--ratings", required=True, help="CSV path with userId,itemId[,rating]")
    p.add_argument("--multiDelim", default=None, help="Delimiter for multi‑item column, e.g. ';'")
    p.add_argument("--userId", help="target user (raw ID) for recommendation")
    p.add_argument("--itemId", help="target item (raw ID) for similarity")
    p.add_argument("--topK", type=int, default=10)
    p.add_argument("--rank", type=int, default=50)
    p.add_argument("--reg", type=float, default=0.01)
    p.add_argument("--iter", type=int, default=10)
    return p.parse_args()


def _run_cli(ns: argparse.Namespace):
    spark = SparkSession.builder.appName("ItemCF_ALS_CLI").getOrCreate()
    df = spark.read.option("header", True).csv(ns.ratings)

    rec = ItemCF_ALS(spark, rank=ns.rank, reg_param=ns.reg, max_iter=ns.iter)
    rec.train(df, multi_delim=ns.multiDelim)

    if ns.itemId:
        sims = rec.get_similar_items(ns.itemId, ns.topK)
        print(f"Top {ns.topK} similar items to {ns.itemId}:")
        for iid, s in sims:
            print(f"  {iid}\t{s:.4f}")
    if ns.userId:
        recs = rec.recommend_for_user(ns.userId, ns.topK)
        print(f"Top {ns.topK} recommendations for user {ns.userId}:")
        for iid, sc in recs:
            print(f"  {iid}\t{sc:.4f}")

    spark.stop()


def _self_test():
    spark = SparkSession.builder.master("local[1]").appName("ItemCF_ALS_TEST").getOrCreate()
    tiny = spark.createDataFrame([
        ("A", "a;b;d"), ("B", "a;c;d"), ("C", "b;e"), ("D", "c;d;e"),
    ], ["userId", "itemId"])
    rec = ItemCF_ALS(spark, rank=4, max_iter=3)
    rec.train(tiny, multi_delim=";")
    print("[TEST] similar to 'a':", rec.get_similar_items("a", 3))
    print("[TEST] recs for 'A':", rec.recommend_for_user("A", 3))
    spark.stop()


if __name__ == "__main__":
    ns = _parse_cli()
    if ns is None:
        _self_test()
    else:
        _run_cli(ns)
