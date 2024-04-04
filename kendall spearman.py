import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 假设df是你的DataFrame，里面有9种数据的对数收益率
df=pd.read_excel('/Users/zhangyichi/Desktop/pythonProject/2023华东杯/对数收益率综合.xlsx',sheet_name='Sheet3')

# 计算Kendall相关系数
kendall_corr = (df.iloc[:,1:]).corr(method='kendall')

# 计算Spearman相关系数
spearman_corr =(df.iloc[:,1:]).corr(method='spearman')

# 绘制Kendall相关系数的热力图
plt.figure(figsize=(12,8))
sns.heatmap(kendall_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("2022 Kendall correlation heatmap")
plt.savefig('/Users/zhangyichi/Desktop/pythonProject/2023华东杯/2022 Kendall correlation heatmap.png')

# 绘制Spearman相关系数的热力图
plt.figure(figsize=(12,8))
sns.heatmap(spearman_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("2022 Spearman correlation heatmap")
plt.savefig('/Users/zhangyichi/Desktop/pythonProject/2023华东杯/2022 Spearman correlation heatmap.png')



