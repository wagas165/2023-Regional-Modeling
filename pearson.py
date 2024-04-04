import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


# 加载数据
df1 = pd.read_excel('/Users/zhangyichi/Desktop/对数收益率综合.xlsx', sheet_name='Sheet1')

x1 = df1['螺纹钢']  # 不锈钢价格
x2 = df1['热轧钢板']  # 铁矿石价格
y=df1['不锈钢']

# 将自变量组合成一个二维数组
X = np.column_stack((x1, x2))

# 创建一个多元线性回归模型
model = LinearRegression()

# 拟合数据
model.fit(X, y)

# 计算R^2得分
r2 = model.score(X, y)

# 输出R^2得分
print('R^2:', r2)

