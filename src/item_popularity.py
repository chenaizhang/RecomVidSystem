import glob
import pandas as pd

class ItemPopularityCalculator:
    """
    只处理宽表格式：
      userId,itemId
      1,61; 307; 528; 592; 868; 976
    计算每个 itemId 被多少不同用户推荐（热度）。
    """
    def __init__(self, input_pattern: str, output_file: str):
        """
        :param input_pattern: glob 模式匹配文件，如 "output/F4.csv" 或 "output/parts-*.csv"
        :param output_file: 保存结果路径，如 "output/F5.csv"
        """
        self.input_pattern = input_pattern
        self.output_file = output_file
        self.df = None
        self.item_pop = None

    def load_data(self):
        """读取所有匹配的文件并合并为一个 DataFrame"""
        files = glob.glob(self.input_pattern)
        if not files:
            raise FileNotFoundError(f"没有匹配到任何文件: {self.input_pattern}")
        dfs = [pd.read_csv(f) for f in files]
        self.df = pd.concat(dfs, ignore_index=True)
        return self

    def explode_items(self):
        """把 itemId 列按分号拆分，并展开为多行"""
        # 1) 拆分
        self.df['item_list'] = self.df['itemId'].str.split(';')
        # 2) 展开
        self.df = self.df.explode('item_list').reset_index(drop=True)
        # 3) 去除空格并重命名列
        self.df['itemId'] = self.df['item_list'].str.strip()
        # 4) 只保留 userId, itemId 两列
        self.df = self.df[['userId', 'itemId']]
        return self

    def compute_popularity(self):
        """按 itemId 分组，统计不同 userId 的数量"""
        self.item_pop = (
            self.df
                .groupby('itemId')['userId']
                .nunique()
                .reset_index(name='user_count')
                .sort_values('user_count', ascending=False)
        )
        return self

    def save(self):
        """将热度结果保存为 CSV"""
        if self.item_pop is None:
            raise ValueError("请先调用 compute_popularity()")
        self.item_pop.to_csv(self.output_file, index=False)
        return self

    def run(self):
        """一键执行完整流程"""
        return (
            self.load_data()
                .explode_items()
                .compute_popularity()
                .save()
        )


if __name__ == '__main__':
    calculator = ItemPopularityCalculator(
        input_pattern='output/Retrieval.csv',    # 或 "output/parts-*.csv"
        output_file='output/F5.csv'
    )
    calculator.run()
    print("已生成 F5.csv，字段说明：")
    print("  itemId      视频 ID")
    print("  user_count  被推荐的不同用户数（热度）")
