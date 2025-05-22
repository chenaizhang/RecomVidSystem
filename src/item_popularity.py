# src/item_popularity.py

import glob
import pandas as pd

class ItemPopularityCalculator:
    """
    针对宽表格式（userId,itemId1;itemId2;...）计算物品热度。
    热度定义为不同用户推荐或交互该物品的用户数量。
    """
    def __init__(self, input_pattern: str, output_file: str):
        """
        初始化 ItemPopularityCalculator。

        参数:
            input_pattern (str): 输入文件的 glob 模式，示例 "output/Retrieval.csv" 或 "output/parts-*.csv"。
            output_file (str): 结果保存路径，示例 "output/F5.csv"。
        """
        self.input_pattern = input_pattern
        self.output_file = output_file
        self.df = None
        self.item_pop = None

    def load_data(self):
        """
        加载并合并所有匹配的 CSV 文件。

        读取 glob 匹配到的文件列表，将它们读取为 DataFrame 并纵向合并。
        返回 self 以支持链式调用。
        """
        files = glob.glob(self.input_pattern)
        if not files:
            raise FileNotFoundError(f"未找到匹配的文件: {self.input_pattern}")
        # 批量读取并合并
        dfs = [pd.read_csv(f) for f in files]
        self.df = pd.concat(dfs, ignore_index=True)
        return self

    def explode_items(self):
        """
        将 itemId 列按分号拆分并展开为多行。

        步骤:
          1. 使用 str.split 将 itemId 拆分为列表，存入临时列 item_list；
          2. 调用 explode 方法将列表展开为多行；
          3. 对 item_list 元素 strip，生成新的 itemId 列；
          4. 丢弃临时列，仅保留 ['userId','itemId']。
        返回 self。
        """
        # 拆分为列表
        self.df['item_list'] = self.df['itemId'].str.split(';')
        # 列表展开
        self.df = self.df.explode('item_list').reset_index(drop=True)
        # 去除空白并重写 itemId
        self.df['itemId'] = self.df['item_list'].str.strip()
        # 保留必要列
        self.df = self.df[['userId', 'itemId']]
        return self

    def compute_popularity(self):
        """
        统计每个 itemId 对应的唯一 userId 数量，作为热度指标。

        返回 self，并将结果保存到 self.item_pop。
        """
        self.item_pop = (
            self.df
            .groupby('itemId')['userId']
            .nunique()
            .reset_index(name='user_count')
            .sort_values('user_count', ascending=False)
        )
        return self

    def save(self):
        """
        将热度统计结果保存为 CSV 文件。

        若未执行 compute_popularity，将抛出异常。
        """
        if self.item_pop is None:
            raise ValueError("请先调用 compute_popularity() 生成数据")
        # 保存到 CSV，不包含索引
        self.item_pop.to_csv(self.output_file, index=False)
        return self

    def run(self):
        """
        一键执行完整流程：加载数据 -> 拆分展开 -> 计算热度 -> 保存结果。

        返回 self。
        """
        return (
            self.load_data()
                .explode_items()
                .compute_popularity()
                .save()
        )

if __name__ == '__main__':
    calc = ItemPopularityCalculator(
        input_pattern='output/Retrieval.csv',  # 或 'output/parts-*.csv'
        output_file='output/F5.csv'
    )
    calc.run()
    print("已生成 F5.csv，字段说明：")
    print("  itemId     视频ID")
    print("  user_count 不同用户推荐数（热度）")
