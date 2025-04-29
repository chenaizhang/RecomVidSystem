import pandas as pd
import numpy as np

# 读取用户视频数据
df = pd.read_csv('Data_backup.csv')

# 转换为 User-Item 表，行是用户，列是视频，值为观看（1表示观看过，0表示未观看）
user_item = {}

# 遍历每个用户的观看记录
for index, row in df.iterrows():
    user_id = row['user']
    video_ids = list(map(int, row['video_ids'].split(';')))
    user_item[user_id] = {video_id: 1 for video_id in video_ids}

# 将用户-物品数据转换为 DataFrame 形式
user_item_df = pd.DataFrame.from_dict(user_item, orient='index').fillna(0)

print("建立 User-Item 表成功！")



# Item-User 倒排表，存储每个视频的用户列表
item_user = {}

for user_id, videos in user_item.items():
    for video_id in videos.keys():
        if video_id not in item_user:
            item_user[video_id] = []
        item_user[video_id].append(user_id)

# 将倒排表转换为 DataFrame
item_user_df = pd.DataFrame.from_dict(item_user, orient='index')

print("建立 Item-User 倒排表成功！")



from collections import defaultdict

def UserInterSection(item_user):
    """
    建立用户物品交集矩阵W, 其中C[u][v]代表的含义是用户u和用户v之间共同喜欢的物品数
    :param item_user: item_user 倒排表
    """
    userInterSection = defaultdict(lambda: defaultdict(int))  # 使用默认字典初始化
    for item, users in item_user.items():
        for u in users:
            for v in users:
                if u == v:
                    continue
                userInterSection[u][v] += 1  # 将用户u和用户v共同喜欢的物品数量加一
    
    return userInterSection

user_intersection_matrix= UserInterSection(item_user)
# 将交集矩阵转换为 DataFrame
user_intersection_df = pd.DataFrame(user_intersection_matrix).fillna(0)
print("建立用户物品交集矩阵成功！")



from sklearn.metrics.pairwise import cosine_similarity

# 计算用户相似度矩阵
cosine_sim = cosine_similarity(user_intersection_df)

# 将相似度矩阵保存到 DataFrame
user_similarity_df = pd.DataFrame(cosine_sim, index=user_item.keys(), columns=user_item.keys())

# 输出用户相似度矩阵
# print(user_similarity_df.head())

print("建立用户相似度矩阵成功！")





K = 5  # 选择最相似的K个用户

def find_top_k_similar_users(user_id, k=5):
    # 获取目标用户与其他用户的相似度
    similarity_scores = user_similarity_df[user_id].sort_values(ascending=False)
    # 排除目标用户本身，选择前K个最相似的用户
    return similarity_scores.drop(user_id).head(k).index.tolist()

# 例如，找到用户 1 的最相似的 5 个用户
similar_users = find_top_k_similar_users(1, k=K)
# print(similar_users)

def calculate_user_interest(user_id, similar_users, user_item, user_similarity_df):
    video_scores = {}
    
    # 计算用户对每个视频的兴趣度
    for similar_user in similar_users:
        similarity = user_similarity_df.loc[user_id, similar_user]
        for video_id in user_item[similar_user].keys():
            if video_id not in video_scores:
                video_scores[video_id] = 0
            video_scores[video_id] += similarity  # 使用相似度作为权重

    # 将视频按兴趣度排序
    ranked_videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_videos

# 例如，计算用户1对视频的兴趣度
ranked_videos_user_1 = calculate_user_interest(1, similar_users, user_item, user_similarity_df)

# 输出前10个视频及其兴趣度
# print(ranked_videos_user_1[:10])



interest_degrees = []

# 对每个用户进行兴趣度计算
for user_id in user_item.keys():
    similar_users = find_top_k_similar_users(user_id, k=K)
    ranked_videos = calculate_user_interest(user_id, similar_users, user_item, user_similarity_df)
    
    # 将用户和视频兴趣度保存到列表
    for video_id, score in ranked_videos:
        interest_degrees.append({'user_id': user_id, 'video_id': video_id, 'interest_degree': score})

# 将兴趣度数据保存为 CSV 文件
interest_df = pd.DataFrame(interest_degrees)
interest_df.to_csv('interest_degree.csv', index=False)
print("用户视频兴趣度计算完成，结果已保存为 interest_degree.csv！")