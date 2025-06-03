import pandas as pd

# === 1. 读取数据 ===
retrieval_df = pd.read_csv("output/Retrieval.csv")
video_info_df = pd.read_csv("output/F6.csv")

# === 2. 数据预处理 ===
retrieval_df['itemId'] = retrieval_df['itemId'].apply(
    lambda x: [int(i.strip()) for i in str(x).split(';') if i.strip().isdigit()]
)

video_meta = video_info_df.set_index('itemId')[['cluster', 'watch']].to_dict('index')

# === 3. 筛选函数===
def filter_recommendations(video_ids):
    result = []
    last_cluster = None
    cluster_streak = 0

    for vid in video_ids:
        if vid not in video_meta:
            continue

        cluster = video_meta[vid]['cluster']
        watch = video_meta[vid]['watch']

        # 控制连续同类不超过5个
        if cluster == last_cluster:
            cluster_streak += 1
        else:
            cluster_streak = 1
            last_cluster = cluster

        if cluster_streak > 5:
            continue

        result.append((vid, cluster, watch))

        if len(result) == 10:
            break

    # 如果不足10个，说明本身推荐就不合格
    if len(result) < 10:
        return [v[0] for v in result]  # 尽力返回已有内容

    clusters = [v[1] for v in result]
    watches = [v[2] for v in result]
    cluster_set = set(clusters)
    has_watch_lt_500 = any(w < 500 for w in watches)

    # Case 1: 替换一个视频以加入 watch < 500
    if not has_watch_lt_500:
        for vid in video_ids[10:]:
            if vid in video_meta and video_meta[vid]['watch'] < 500:
                new_cluster = video_meta[vid]['cluster']
                new_watch = video_meta[vid]['watch']
                max_idx = max(range(10), key=lambda i: result[i][2])
                result[max_idx] = (vid, new_cluster, new_watch)
                break  # 替换一次即可
        # 更新检查值
        clusters = [v[1] for v in result]
        watches = [v[2] for v in result]
        cluster_set = set(clusters)
        has_watch_lt_500 = any(w < 500 for w in watches)

    # Case 2: 替换一个视频以增加类别数量
    if len(cluster_set) < 3:
        for vid in video_ids[10:]:
            if vid in video_meta:
                new_cluster = video_meta[vid]['cluster']
                new_watch = video_meta[vid]['watch']
                if new_cluster not in cluster_set:
                    for i in range(10):
                        replaced = result.copy()
                        replaced[i] = (vid, new_cluster, new_watch)
                        new_clusters = [v[1] for v in replaced]
                        if len(set(new_clusters)) >= 3:
                            result = replaced
                            cluster_set = set(new_clusters)
                            watches = [v[2] for v in result]
                            has_watch_lt_500 = any(w < 500 for w in watches)
                            break
                    break  # 替换一次即可

    # 最终：不管是否满足条件，都返回当前的10个结果（尽力推荐）
    return [v[0] for v in result]

# === 4. 执行推荐筛选 ===
filtered_recommendations = []
for _, row in retrieval_df.iterrows():
    uid = row['userId']
    print(f"正在处理用户 userId = {uid}")
    filtered_items = filter_recommendations(row['itemId'])
    if filtered_items:
        filtered_recommendations.append({
            'userId': uid,
            'itemId': '; '.join(map(str, filtered_items))
        })

# === 5. 保存结果 ===
final_df = pd.DataFrame(filtered_recommendations)
final_df.to_csv("output/Filtered_Recommendations.csv", index=False)
print("筛选完成，结果已保存为 Filtered_Recommendations.csv")
