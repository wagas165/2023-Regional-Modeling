
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 读取Excel文件
df1 = pd.read_excel('/Users/zhangyichi/Desktop/对数收益率综合.xlsx', sheet_name='Sheet3')

# 定义商品分类
products = ['螺纹钢', '热轧钢板', '不锈钢', '线材', '铁矿石', '焦煤', '焦炭', '锰硅', '硅铁']

# 初始化空列表
best_lags = []
pearson_correlations = []

# 计算交叉相关值
total_length=len(df1['螺纹钢'])
for i in range(len(products)):
    for j in range(len(products)):
        if i == j:
            continue
        price_i = df1[products[i]]
        price_j = df1[products[j]]
        max_lags = 30
        pearson_correlations_for_max_lag = []
        for max_lag in range(1, max_lags + 1):
            pearson_correlations_for_lag = []
            for lag in range(0, max_lag):
                shifted_price_j = price_j.shift(lag)
                valid_price_i = price_i[lag:]
                valid_shifted_price_j = (shifted_price_j.dropna())
                correlation, _ = pearsonr(valid_price_i, valid_shifted_price_j)
                pearson_correlations_for_lag.append(correlation)


            abs_pearson = [abs(x) for x in pearson_correlations_for_lag]
            best_lag_for_max_lag = np.argmax(abs_pearson)
            pearson_correlations_for_max_lag.append(pearson_correlations_for_lag[best_lag_for_max_lag])
        abs_pearson_for_max_lag = [abs(x) for x in pearson_correlations_for_max_lag]
        best_lag = np.argmax(abs_pearson_for_max_lag)
        best_lags.append(best_lag)
        pearson_correlations.append(pearson_correlations_for_max_lag[best_lag])
# 输出结果
k=0
for i in range(len(products)):
    for j in range(len(products)):
        if i == j:
            continue
        print(f'{products[i]} vs. {products[j]}: best lag = {best_lags[k]}, pearson correlation = {pearson_correlations[k]}')
        k+=1

# 将相关系数数组转换成矩阵形式
k=0
corr_mat = np.zeros((len(products), len(products)))
for i in range(len(products)):
    for j in range(len(products)):
        if i == j:
            continue
        corr_mat[i, j] = pearson_correlations[k]
        k+=1

# 绘制热图
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_mat, cmap='seismic', vmin=-1, vmax=1)

# 添加坐标轴标签
ax.set_xticks(np.arange(len(products)))
ax.set_yticks(np.arange(len(products)))
ax.set_xticklabels(products)
ax.set_yticklabels(products)

# 添加每个格子内的数值
for i in range(len(products)):
    for j in range(len(products)):
        if not np.isnan(corr_mat[i, j]):
            text = ax.text(j, i, f'{corr_mat[i, j]:.2f}', ha='center', va='center', color='black')

# 添加颜色条
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Pearson correlation coefficient', rotation=-90, va='bottom')

# 添加标题
ax.set_title('Pearson Correlation Coefficients of Commodity Prices')

# 保存热图
if not os.path.exists('/Users/zhangyichi/Desktop/商品价格相关性热图'):
    os.makedirs('/Users/zhangyichi/Desktop/商品价格相关性热图')
plt.savefig('/Users/zhangyichi/Desktop/商品价格相关性热图/商品价格相关性热图2022.png', dpi=300, bbox_inches='tight')
plt.show()

# 将最佳滞后数组转换成矩阵形式
k=0
lag_mat = np.zeros((len(products), len(products)))
for i in range(len(products)):
    for j in range(len(products)):
        if i == j:
            continue
        lag_mat[i, j] = best_lags[k]
        k+=1

# 绘制滞后期热图
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(lag_mat, cmap='Reds', vmin=0, vmax=max_lags)

# 添加坐标轴标签
ax.set_xticks(np.arange(len(products)))
ax.set_yticks(np.arange(len(products)))
ax.set_xticklabels(products)
ax.set_yticklabels(products)

# 添加每个格子内的数值
for i in range(len(products)):
    for j in range(len(products)):
        text = ax.text(j, i, f'{lag_mat[i, j]:.0f}', ha='center', va='center', color='black')

# 添加颜色条
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Best lag', rotation=-90, va='bottom')

# 添加标题
ax.set_title('Best Lags between Commodity Prices')

# 保存热图
if not os.path.exists('/Users/zhangyichi/Desktop/商品价格相关性热图'):
    os.makedirs('/Users/zhangyichi/Desktop/商品价格相关性热图')
plt.savefig('/Users/zhangyichi/Desktop/商品价格相关性热图/商品价格滞后期热图2022.png', dpi=300, bbox_inches='tight')
plt.show()




