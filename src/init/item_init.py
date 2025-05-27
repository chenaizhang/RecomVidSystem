import os
import pandas as pd
import random

def parse_duration(dur_str):
    """
    将“分钟:秒”格式的字符串转换为总秒数。
    例如 "3:45" -> 225
    """
    try:
        minutes, seconds = dur_str.split(':')
        return int(minutes) * 60 + int(seconds)
    except:
        return pd.NA

def main():
    # 文件路径
    user_path = 'data/original/user_orig.csv'
    item_path = 'data/original/item_orig.csv'
    output_path = 'data/item.csv'

    # 读取数据
    users = pd.read_csv(user_path)
    items = pd.read_csv(item_path)

    # 1. 预处理 item 表，映射并重命名列
    items['length']  = items['视频时长'].astype(str).apply(parse_duration)
    items['comment'] = items['弹幕数']
    items['like']    = items['点赞数']
    items['watch']   = items['播放量']
    items['share']   = items['收藏量']
    items['name']    = items['标题']
    # 保留需要的列 + 分区类型
    items = items[['标题','length','comment','like','watch','share','name','分区类型']].dropna(subset=['length','name'])
    # 添加 itemId
    items.insert(0, 'itemId', range(1, len(items) + 1))

    # 2. 计算每个分区的总播放量及比例
    total_watch = items['watch'].sum()
    cat_watch = items.groupby('分区类型')['watch'].sum()
    cat_prop  = cat_watch / total_watch

    # 3. 按分区随机分配用户
    all_user_ids = users['id'].tolist()
    total_users  = len(all_user_ids)
    random.seed(42)

    category_users = {}
    for cat, prop in cat_prop.items():
        n = round(prop * total_users * 100)
        n = min(n, total_users)
        if n <= 0:
            continue
        if n <= total_users:
            category_users[cat] = random.sample(all_user_ids, n)
        else:
            category_users[cat] = random.choices(all_user_ids, k=n)

    # 4. 按视频再次分配用户
    records = []
    for _, row in items.iterrows():
        cat = row['分区类型']
        users_in_cat = category_users.get(cat, [])
        sum_watch_in_cat = cat_watch.get(cat, 0)

        # 计算本视频应分配的用户数
        if sum_watch_in_cat > 0 and users_in_cat:
            ratio = row['watch'] / sum_watch_in_cat
            # 注意乘 100 以减小稀疏性
            k = round(len(users_in_cat) * 100 * ratio)
            # 限制人数在 [20, 300]
            k = max(20, min(k, 300))
        else:
            k = 0

        # 随机抽样
        if k > 0:
            if k <= len(users_in_cat):
                assigned = random.sample(users_in_cat, k)
            else:
                assigned = random.choices(users_in_cat, k=k)
        else:
            assigned = []

        # 生成记录
        records.append({
            'itemId':  row['itemId'],
            'length':  row['length'],
            'comment': row['comment'],
            'like':    row['like'],
            'watch':   row['watch'],
            'share':   row['share'],
            'name':    row['name'],
            'userId':  '; '.join(map(str, assigned))
        })

    # 5. 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'已保存到 {output_path}，共 {len(records)} 条记录。')

if __name__ == '__main__':
    main()
