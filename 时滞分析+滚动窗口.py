import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

def rolling_corr(window_data, shifted_data):
    if len(window_data) == len(shifted_data) and len(window_data) > 0:
        return np.corrcoef(window_data, shifted_data)[0, 1]
    else:
        return None

df1 = pd.read_excel('/Users/zhangyichi/Desktop/对数收益率综合.xlsx', sheet_name='Sheet3')

products = ['螺纹钢', '热轧钢板', '不锈钢', '线材', '铁矿石', '焦煤', '焦炭', '锰硅', '硅铁']
window_size = 15
max_lag = 30

rolling_correlations = pd.DataFrame(columns=products, index=df1.index)
rolling_best_lags = pd.DataFrame(columns=products, index=df1.index)

for i in range(len(products)):
    for j in range(i + 1, len(products)):
        product_i = df1[products[i]]
        product_j = df1[products[j]]

        best_corrs = []
        best_lags = []
        for t in range(len(product_i) - window_size + 1):
            window_data = product_i.iloc[t: t + window_size]
            best_corr = None
            best_lag = None
            for lag in range(max_lag + 1):
                shifted_data = product_j.shift(lag).dropna().iloc[t: t + window_size]
                corr = rolling_corr(window_data, shifted_data)

                if corr is not None:
                    if best_corr is None or abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag
            best_corrs.append(best_corr)
            best_lags.append(best_lag)

        rolling_correlations.loc[window_size - 1 :, f'{products[i]}-{products[j]}'] = best_corrs
        rolling_best_lags.loc[window_size - 1 :, f'{products[i]}-{products[j]}'] = best_lags
# # 设置最大显示行数
# pd.set_option('display.max_rows', None)
#
# # 设置最大显示列数
# pd.set_option('display.max_columns', None)
#
# # 设置最大显示宽度
# pd.set_option('display.width', None)
#
# print(rolling_best_lags)
for i in range(9):
    rolling_correlations_clean = rolling_correlations.iloc[:, (9+4*i):(13+4*i)].dropna()
    rolling_correlations_clean.plot(figsize=(15, 8))
    plt.title('Rolling Correlations between Commodity Prices in 2022 (15-day window)')
    plt.ylabel('Correlation coefficient')
    plt.xlabel('Days')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'/Users/zhangyichi/Desktop/商品价格相关性热图/时滞分析+滚动窗口{window_size}天 2022（{i+1}）.png')
    plt.show()



