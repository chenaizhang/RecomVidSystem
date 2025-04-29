import pandas as pd

# 读取 Data_user.csv
data = pd.read_csv('Data_user.csv')

# 创建用户-视频矩阵
video_ids = set()
for video_list in data['video_ids']:
    video_ids.update(video_list.split(';'))

video_ids = list(video_ids)  # 将所有视频ID列表化
user_video_matrix = pd.DataFrame(0, index=data['user'], columns=video_ids)

# 填充用户观看记录
for idx, row in data.iterrows():
    user = row['user']
    watched_videos = row['video_ids'].split(';')
    user_video_matrix.loc[user, watched_videos] = 1

# 显示用户-视频矩阵的一部分
print(user_video_matrix.head())

from sklearn.metrics.pairwise import cosine_similarity

# 计算用户相似度矩阵
similarity_matrix = cosine_similarity(user_video_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=user_video_matrix.index, columns=user_video_matrix.index)

# 查看部分用户相似度
print(similarity_df.head())

# 计算每个用户对每个视频的兴趣度
interest_degree = pd.DataFrame(0.0, index=user_video_matrix.index, columns=user_video_matrix.columns)

for user in user_video_matrix.index:
    # 获取当前用户的观看记录
    print(f"Calculating interest degree for user: {user}") 

    # 获取与当前用户相似的用户
    similar_users = similarity_df[user].sort_values(ascending=False)[1:]  # 排除自己
    total_similarity = similar_users.sum()

    # 为每个视频计算兴趣度
    for video in user_video_matrix.columns:
        print(f"Calculating interest degree for video: {video}")

        weighted_sum = 0
        for similar_user, similarity in similar_users.items():
            if user_video_matrix.loc[similar_user, video] == 1:  # 如果相似用户观看了该视频
                weighted_sum += similarity

        # 计算兴趣度
        if total_similarity > 0:
            interest_degree.loc[user, video] = weighted_sum / total_similarity

# 输出部分兴趣度
print(interest_degree.head())

