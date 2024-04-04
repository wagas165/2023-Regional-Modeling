import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

df=pd.read_excel("/Users/zhangyichi/Desktop/2022data 对数收益率.xlsx",sheet_name='不锈钢')

need=df['对数收益率'].iloc[1:]
sm.qqplot(need,line='s')
plt.show()