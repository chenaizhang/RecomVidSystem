import pandas as pd  # 数据处理库
import numpy as np  # 数值计算库


def process_user_data():
    """
    处理原始用户数据，包含数据清洗和特征生成步骤
    
    流程说明:
        1. 读取原始CSV文件
        2. 删除无用的level列
        3. 生成符合正态分布的用户年龄
        4. 重新编号id列保证唯一性
        5. 保存处理后的数据
    """
    # 步骤1：读取原始数据（编码为gb18030处理中文）
    df = pd.read_csv('user_info.csv', encoding='gb18030')

    # 步骤2：删除用户等级列（假设业务需求不需要该字段）
    if 'level' in df.columns:
        df.drop('level', axis=1, inplace=True)  # axis=1表示删除列

    # 步骤3：生成用户年龄（模拟真实用户年龄分布）
    mu, sigma = 37, 10  # 均值37岁，标准差10（覆盖主要用户群体）
    age = np.random.normal(mu, sigma, len(df))  # 生成正态分布随机数
    # 截断年龄范围（14-60岁更符合实际用户情况）并转为整数
    age = np.clip(age, 14, 60).astype(int)
    df['age'] = age  # 添加新的年龄列

    # 步骤4：重新编号id列（原始id可能不连续或重复）
    df['id'] = range(1, len(df)+1)  # 从1开始连续编号

    # 步骤5：保存处理后的数据（utf-8编码通用）
    df.to_csv('user.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    process_user_data()  # 执行数据处理主函数