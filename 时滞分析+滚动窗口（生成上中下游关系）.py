import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# # 设置最大显示行数
# pd.set_option('display.max_rows', None)
#
# # 设置最大显示列数
# pd.set_option('display.max_columns', None)
#
# # 设置最大显示宽度
# pd.set_option('display.width', None)


plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

def rolling_corr(window_data, shifted_data):
    if len(window_data) == len(shifted_data) and len(window_data) > 0:
        return np.corrcoef(window_data, shifted_data)[0, 1]
    else:
        return None

df1 = pd.read_excel('/Users/zhangyichi/Desktop/对数收益率综合.xlsx', sheet_name='Sheet3')

upstream = ['铁矿石', '焦煤', '焦炭']
midstream = ['锰硅', '硅铁']
downstream = ['热轧钢板', '螺纹钢', '线材', '不锈钢']
combinations = [
    (upstream, midstream, "Upstream-Midstream"),
    (midstream, downstream, "Midstream-Downstream"),
    (upstream, downstream, "Upstream-Downstream"),
]

window_size = 45
max_lag = 30

df_tosave=pd.DataFrame()

for combination in combinations:
    stage1, stage2, title = combination
    rolling_correlations = pd.DataFrame(columns=stage2, index=df1.index)

    for product_i_name in stage1:
        for product_j_name in stage2:
            product_i = df1[product_i_name]
            product_j = df1[product_j_name]

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

            rolling_correlations.loc[window_size - 1:, f'{product_i_name}-{product_j_name}'] = best_corrs
    rolling_correlations_clean = rolling_correlations.iloc[:, -len(stage1)*len(stage2):].dropna()
    rolling_correlations_clean.plot(figsize=(15, 8))
    plt.title(f'Rolling Correlations between {title} Commodity Prices in 2022 (45-day window)')
    plt.ylabel('Correlation coefficient')
    plt.xlabel('Days')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'/Users/zhangyichi/Desktop/商品价格相关性热图/{title} 时滞分析+滚动窗口{window_size}天 2022.png')
    plt.show()
    rolling_correlations_clean.to_excel(f'/Users/zhangyichi/Desktop/商品价格相关性数据/{title} 时滞分析+滚动窗口{window_size}天 2022.xls')


