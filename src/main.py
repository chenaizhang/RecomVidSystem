import numpy as np
import os
import csv
import pandas as pd
import time
from usercf import LoadDataFromCSV, UserCF, save_recommendperson_results
from itemcf_wals import WALSRecommender
from item_popularity import ItemPopularityCalculator
from cluster_analysis import ClusteringPipeline

if __name__ == "__main__":
    # F3 为特定用户推荐相似兴趣的用户群体。
    # 加载训练数据
    train = LoadDataFromCSV("data/Data_ui.csv")
    print(f"train data size: {len(train)}")

    # 训练模型
    start = time.time()
    model = UserCF(train)
    model.train()
    print(f"模型训练时间: {time.time() - start:.2f} 秒")

    # 测试用户列表（前 5 个）
    test_users = list(train.keys())

    # 保存相似用户推荐结果
    save_recommendperson_results("output/F3.csv", model, test_users, N=10)
    print("相似用户推荐结果已保存到 output/F3.csv")

    # # 打印示例结果
    # for user in test_users:
    #     similar = model.recommendperson(user, 10)
    #     print(f"Users similar to {user}: {similar}")
    # for user in test_users:
    #     recs = model.recommend(user, 5, 80)
    #     print(f"Recommendations for {user}: {recs}")

    print("--- " * 10)

    # F4 根据用户历史浏览记录，为其推荐相关视频。
    print("正在读取数据...")
    data = pd.read_csv('data/Data_ui.csv')
    recommender = WALSRecommender(num_factors=32)
    print("正在训练模型...")
    recommender.train(data, num_iterations=20, learning_rate=0.001)
    print("正在生成并保存所有用户推荐...")
    recommender.recommend_all_and_save(top_n=5, output_file='output/F4.csv')
    print("正在生成并保存宽表检索结果...")
    recommender.save_retrieval(top_n=100, output_file='output/Retrieval.csv')



    # F5 预测视频未来的观看热度变化。
    calculator = ItemPopularityCalculator(
        input_pattern='output/Retrieval.csv',    # 或 "output/parts-*.csv"
        output_file='output/F5.csv'
    )
    calculator.run()
    print("已生成 F5.csv，字段说明：")
    print("  itemId      视频 ID")
    print("  user_count  被推荐的不同用户数（热度）")

    
    
    # F6 对视频进行聚类，找出具有相似观看用户的视频。
    # F7 对用户进行聚类，找出具有相似观看兴趣的用户群体。
    pipeline = ClusteringPipeline(
        user_input_path='data/user.csv',
        item_input_path='data/item.csv',
        output_dir='output',
        n_user_clusters=4,
        n_item_clusters=30
    )
    pipeline.run()