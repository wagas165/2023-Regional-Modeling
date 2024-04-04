import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import pearsonr

df1 = pd.read_excel('/Users/zhangyichi/Desktop/对数收益率综合.xlsx', sheet_name='Sheet1')


products = ['螺纹钢', '热轧钢板', '不锈钢', '线材', '铁矿石', '焦煤', '焦炭', '锰硅', '硅铁']
product_data = { }
for i,name in enumerate(products):
    product_data[i]=df1[name]

# 设置窗口大小范围
min_window_size = 8
max_window_size = 50

# 初始化一个空字典来存储窗口期信息
window_periods = {}

# 生成所有可能的产品组合
products = list(product_data.keys())
product_combinations = list(combinations(products, 2))
product_combinations = [(products[i], products[j]) for (i,j) in product_combinations]

# 遍历所有组合
for product_A, product_B in product_combinations:
    price_A = product_data[product_A]
    price_B = product_data[product_B]

    best_window_size = None
    best_correlation = -1

    # 尝试不同的窗口大小
    for window_size in range(min_window_size, max_window_size + 1):
        correlations = []
        for i in range(len(price_A) - window_size + 1):
            window_price_A = price_A.iloc[i:i + window_size]
            window_price_B = price_B.iloc[i:i + window_size]

            valid_indices = (~np.isnan(window_price_A)) & (~np.isnan(window_price_B))
            valid_window_price_A = window_price_A[valid_indices]
            valid_window_price_B = window_price_B[valid_indices]

            correlation, _ = pearsonr(valid_window_price_A, valid_window_price_B)
            correlations.append(abs(correlation))

        avg_correlation = np.mean(correlations)

        # 更新最佳窗口大小和相关性
        if avg_correlation > best_correlation:
            best_correlation = avg_correlation
            best_window_size = window_size

    # 将结果存储在字典中
    products = ['螺纹钢', '热轧钢板', '不锈钢', '线材', '铁矿石', '焦煤', '焦炭', '锰硅', '硅铁']
    window_periods[(products[product_A], products[product_B])] = best_window_size

# 输出窗口期信息
print(window_periods)
