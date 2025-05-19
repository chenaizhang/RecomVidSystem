# generate_recs.py -----------------------------------------------------------
from itemcf_als import ItemCF_ALS
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, explode, split, trim, lit, row_number,
    concat_ws, format_number, collect_list, udf,
)
import argparse

# ------------------------- CLI ---------------------------------------------
p = argparse.ArgumentParser("Bulk recommend (Top-K, no history, with score)")
p.add_argument("--ratings", required=True)
p.add_argument("--multiDelim", default=None)
p.add_argument("--topK", type=int, default=10)
p.add_argument("--rank", type=int, default=50)
p.add_argument("--reg", type=float, default=0.01)
p.add_argument("--iter", type=int, default=10)
p.add_argument("--output", default="output.csv")
args = p.parse_args()

spark = SparkSession.builder.appName("BulkItemCF").getOrCreate()

# 1) 读原始交互
df_raw = spark.read.option("header", True).csv(args.ratings)

# 2) 训练
rec = ItemCF_ALS(spark, rank=args.rank, reg_param=args.reg, max_iter=args.iter)
rec.train(df_raw, multi_delim=args.multiDelim)

# 3) recommendForAllUsers 取多一点
raw_k = args.topK * 5
recs = rec.model.recommendForAllUsers(raw_k)

# 4) >>> 历史 explode + trim  <<<  防遗漏
hist_raw = df_raw
if args.multiDelim:
    hist_raw = hist_raw.withColumn("itemId", explode(split(col("itemId"), args.multiDelim)))
hist_raw = hist_raw.withColumn("itemId", trim(col("itemId")))

# 映射到索引
u_map_df = spark.createDataFrame([(i, u) for i, u in rec.user_map.id2raw.items()],
                                 ["userIdx", "userId"])
i_map_df = spark.createDataFrame([(i, it) for i, it in rec.item_map.id2raw.items()],
                                 ["itemIdx", "itemId"])
hist = (hist_raw.join(u_map_df, "userId")
                 .join(i_map_df, "itemId")
                 .select("userIdx", "itemIdx"))

# 5) explode 推荐并去历史
exploded = (recs.select("userIdx", explode("recommendations").alias("rec"))
                 .select("userIdx", col("rec.itemIdx"), col("rec.rating").alias("score")))

filtered = exploded.join(hist, ["userIdx", "itemIdx"], "left_anti")

# 6) 取每用户前 K
w = Window.partitionBy("userIdx").orderBy(col("score").desc())
ranked = filtered.withColumn("rank", row_number().over(w)).filter(col("rank") <= args.topK)

# 7) itemIdx→原始 ID + 格式化 score
idx_to_item = spark.sparkContext.broadcast(rec.item_map.id2raw)
@udf("string")
def idx_to_raw(i): return idx_to_item.value[int(i)]

ranked = ranked.withColumn("itemId", idx_to_raw("itemIdx")) \
               .withColumn("pair", concat_ws(":", "itemId", format_number("score", 4)))

# 8) 合并为分号分隔串
result = (ranked.groupBy("userIdx")
                  .agg(concat_ws(";", collect_list("pair")).alias("itemId_score_list"))
                  .join(u_map_df, "userIdx")
                  .select("userId", "itemId_score_list"))

# 9) 保存
(result.coalesce(1)
       .write.mode("overwrite").option("header", True)
       .csv(args.output))

print(f"✔ 写入 {args.output} 成功，每用户 {args.topK} 条推荐、附带得分，且不含历史观看内容。")
spark.stop()
# ---------------------------------------------------------------------------
