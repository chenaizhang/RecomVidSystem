import glob, pandas as pd

# 1) 读取原始推荐结果  
part_path = glob.glob("output.csv/part-*.csv")[0]   # 只取第一个 part
df = pd.read_csv(part_path)

# 2. 将 itemId_score_list 拆开成多行多列  
#    先把 42836:0.9963;43418:0.9963 … 这样的字段切割成列表
df['item_list'] = df['itemId_score_list'].str.split(';')

# 3. 把列表“打散”为行，每行一个 (userId, itemId:score)  
df = df.explode('item_list').reset_index(drop=True)

# 4. 只保留 itemId，并去掉可能的空格  
df['itemId'] = df['item_list'].str.split(':').str[0].str.strip()

# 5. 统计每个 itemId 被多少不同 userId 推荐到  
item_pop = (
    df.groupby('itemId')['userId']          # 对 itemId 分组
      .nunique()                           # 统计唯一 userId 数
      .reset_index(name='user_count')      # 重命名为热度数
      .sort_values('user_count', ascending=False)  # 按热度降序
)

# 6. 保存结果  
item_pop.to_csv('item_popularity.csv', index=False)

print('已生成 item_popularity.csv，字段说明：')
print('itemId      视频ID')
print('user_count  该视频被推荐给的用户数 = 热度数')
