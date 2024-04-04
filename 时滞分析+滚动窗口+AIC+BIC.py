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

def calculate_aic_bic(n, mse, k):
    aic = n * np.log(mse) + 2 * k
    bic = n * np.log(mse) + np.log(n) * k
    return aic, bic

df1 = pd.read_excel('/Users/zhangyichi/Desktop/对数收益率综合.xlsx', sheet_name='Sheet3')

products = ['螺纹钢', '热轧钢板', '不锈钢', '线材', '铁矿石', '焦煤', '焦炭', '锰硅', '硅铁']
max_lag = 30

window_sizes = list(range(10, 90 ,5))  # 设置窗口期范围，从10天到30天
aic_bic_results = pd.DataFrame(columns=['window_size', 'aic', 'bic'])

for window in window_sizes:
    window_size = window

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

    # 计算MSE
    mse_total = 0
    for i in range(9):
        rolling_correlations_clean = rolling_correlations.iloc[:, (9+4*i):(13+4*i)].dropna()
        mse = ((rolling_correlations_clean - rolling_correlations_clean.mean()) ** 2).mean().mean()
        mse_total += mse

    mse_total = mse_total / 9
    aic, bic = calculate_aic_bic(len(product_i) - window_size + 1, mse_total, 9)

    aic_bic_results = aic_bic_results.append({'window_size': window, 'aic': aic, 'bic': bic}, ignore_index=True)

print(aic_bic_results)

# 找到AIC和BIC值最小的窗口期
optimal_window_aic = aic_bic_results.loc[aic_bic_results['aic'].idxmin()]['window_size']
optimal_window_bic = aic_bic_results.loc[aic_bic_results['bic'].idxmin()]['window_size']

print(f"Optimal window size based on AIC: {optimal_window_aic}")
print(f"Optimal window size based on BIC: {optimal_window_bic}")

# 画出AIC和BIC随窗口期变化的趋势图
plt.figure(figsize=(15, 8))
plt.plot(aic_bic_results['window_size'], aic_bic_results['aic'], marker='o', label='AIC')
plt.plot(aic_bic_results['window_size'], aic_bic_results['bic'], marker='o', label='BIC')
plt.xlabel('Window Size')
plt.ylabel('Information Criterion Value')
plt.title('AIC and BIC for Different Window Sizes')
plt.legend()
plt.savefig('/Users/zhangyichi/Desktop/AIC_BIC窗口期分析.png')
plt.show()





