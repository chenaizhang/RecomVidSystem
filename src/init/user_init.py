import os
import pandas as pd
import random
from collections import defaultdict

def build_user_items_map(item_do_path: str) -> dict:
    """
    从 item_do.csv 中读取 item-user 映射，构建 userId -> [itemId, ...] 的倒排表。
    """
    df = pd.read_csv(item_do_path, dtype={'userId': str})
    user_map = defaultdict(list)
    for _, row in df.iterrows():
        item_id = row['itemId']
        # 拆分 userId 列，可能类似 "1;2;3"
        for uid in str(row['userId']).split(';'):
            uid = uid.strip()
            if uid:
                user_map[uid].append(str(item_id))
    return user_map

def normalize_gender(sex: str) -> str:
    """
    男 -> M，女 -> F，其它（如 保密）随机 M/F
    """
    if sex == '男':
        return 'M'
    elif sex == '女':
        return 'F'
    else:
        return random.choice(['M', 'F'])

def main():
    # 路径设置
    user_in  = 'data/original/user_orig.csv'
    item_do  = 'data/item.csv'
    user_out = 'data/user.csv'

    # 保证输出目录存在
    os.makedirs(os.path.dirname(user_out), exist_ok=True)

    # 构建倒排表
    user_items = build_user_items_map(item_do)

    # 读取用户表
    df = pd.read_csv(user_in, dtype={'id': str})

    # 重命名并映射列
    df = df.rename(columns={'id': 'userId', 'name': 'name', 'following': 'following',
                            'fans': 'fans', 'age': 'age'})
    # 性别映射
    random.seed(42)
    df['gender'] = df['sex'].apply(normalize_gender)

    # 根据倒排表添加 itemId 列，并丢弃无映射的行
    def make_item_list(uid):
        items = user_items.get(uid, [])
        return '; '.join(items) if items else pd.NA

    df['itemId'] = df['userId'].apply(make_item_list)
    df = df.dropna(subset=['itemId'])

    # 最终列顺序
    df_out = df[['userId', 'itemId', 'name', 'gender', 'following', 'age', 'fans']]

    # 导出
    df_out.to_csv(user_out, index=False, encoding='utf-8-sig')
    print(f'已生成 {user_out}，共 {len(df_out)} 条记录。')

if __name__ == '__main__':
    main()
