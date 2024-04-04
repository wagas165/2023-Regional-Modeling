import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# 加载数据
df1=pd.read_excel('/Users/zhangyichi/Desktop/对数收益率综合.xlsx',sheet_name='Sheet1')

price_A=df1['铁矿石'] #不锈钢价格
price_B=df1['焦炭'] #铁矿石价格

# 初始化空列表
max_lags = range(1, 100)
best_lags = []
pearson_correlations = []

# 计算交叉相关值
for max_lag in max_lags:
    pearson_correlations_for_max_lag = []
    for lag in range(1, max_lag + 1):
        shifted_price_B = price_B.shift(lag)
        # 移除NaN值
        valid_price_A = price_A[lag:]
        valid_shifted_price_B = shifted_price_B.dropna()
        correlation, _ = pearsonr(valid_price_A, valid_shifted_price_B)
        pearson_correlations_for_max_lag.append(correlation)
    # 找到最佳时滞和对应的交叉相关值
    abs_pearson=[abs(x) for x in pearson_correlations_for_max_lag]
    best_lag_for_max_lag = np.argmax(abs_pearson)
    best_lags.append(best_lag_for_max_lag)
    pearson_correlations.append(pearson_correlations_for_max_lag[best_lag_for_max_lag])

# 绘制二维图
fig, ax = plt.subplots()
sc = ax.scatter(max_lags, best_lags,c=pearson_correlations,cmap='seismic')
ax.set_xlabel('largest lag')
ax.set_ylabel('best lag')
cbar = fig.colorbar(sc,ax=ax)
cbar.ax.set_ylabel('pearson coefficient')
plt.title('k')
plt.show()
print(pearson_correlations)





