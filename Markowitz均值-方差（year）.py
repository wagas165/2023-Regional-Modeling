import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


# 定义类
class Markowitz:
    def __init__(self, returns):
        self.returns = returns

    # 求解最小方差组合
    def solveMinVar(self, goal_ret):
        cov = np.cov(self.returns)
        mean = np.mean(self.returns, axis=1)
        row1 = np.concatenate((cov.T, [mean, np.ones_like(mean)]), axis=0).T
        row2 = np.concatenate((np.ones_like(mean), [0, 0]))
        row3 = np.concatenate((mean, [0, 0]))
        A = np.vstack((row1, [row2, row3]))
        b = np.hstack((np.zeros_like(mean), [1, goal_ret]))
        results = linalg.solve(A, b)
        return np.vstack((np.arange(len(mean)), results[:-2]))

    # 计算投资组合方差
    def calVar(self, portion):
        return np.dot(np.dot(portion, np.cov(self.returns)), portion.T)

    # 绘制有效前沿
    def plotFrontier(self):
        goal_ret = np.arange(-0.005, 0.01, 0.0001)
        variance = list(map(lambda x: self.calVar(self.solveMinVar(x)[1, :]), goal_ret))
        plt.plot(variance, goal_ret)
        plt.xlabel('Variance')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')



df = pd.read_excel("/Users/zhangyichi/Desktop/2020SVM.xlsx")

# 替换缺失值为 0
df.fillna(value=0, inplace=True)

# 选择要读取的列
columns_to_read = ['螺纹钢', '热轧卷板', '不锈钢', '线材', '焦炭', '焦煤', '铁矿石', '锰硅', '硅铁']
# 选择要读取的列
columns_to_read = ['螺纹钢', '热轧卷板', '不锈钢', '线材', '焦炭', '焦煤', '铁矿石', '锰硅', '硅铁']
column_data_1 = df[columns_to_read[0]]
column_data_2 = df[columns_to_read[1]]
column_data_3 = df[columns_to_read[2]]
column_data_4 = df[columns_to_read[3]]
column_data_5 = df[columns_to_read[4]]
column_data_6 = df[columns_to_read[5]]
column_data_7 = df[columns_to_read[6]]
column_data_8 = df[columns_to_read[7]]
column_data_9 = df[columns_to_read[8]]

# 将数据存储为序列
data_list_1 = column_data_1.tolist()
data_list_2 = column_data_2.tolist()
data_list_3 = column_data_3.tolist()
data_list_4 = column_data_4.tolist()
data_list_5 = column_data_5.tolist()
data_list_6 = column_data_6.tolist()
data_list_7 = column_data_7.tolist()
data_list_8 = column_data_8.tolist()
data_list_9 = column_data_9.tolist()


# 选择需要的数据
x1 = np.array(data_list_1)
x2 = np.array(data_list_3)
x3 = np.array(data_list_4)
x4 = np.array(data_list_8)
x5 = np.array(data_list_9)

returns = [x1,x2,x3,x4,x5]
# 创建Markowitz对象，并对数据进行分析
m = Markowitz(returns)
m.plotFrontier()


# 计算最小方差组合及其方差
min_var_weights = m.solveMinVar(0)[1, :]
min_var_portfolio_var = m.calVar(min_var_weights)


# 输出最小方差组合和方差
print("最小方差组合权重: {}".format(min_var_weights))
print("最小方差组合方差: {:.6f}".format(min_var_portfolio_var))
plt.title('Efficient Frontier (Min. variance portfolio var = {:.6f})'.format(min_var_portfolio_var))

# 用scipy.optimize.minimize求最小值
def minimizeVolatility(returns):
    cov = np.cov(returns)
    n = len(cov)
    init_guess = np.ones(n)/n
    bounds = tuple((0,1) for asset in range(n))
    weights_sum_to_1 = {'type':'eq', 'fun': lambda weights: np.sum(weights) - 1}
    def objective(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    weights = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=(weights_sum_to_1))
    return weights.x

# 计算有效前沿
def efficientFrontier(returns):
    n = len(returns)
    target_returns = np.linspace(np.min(returns), np.max(returns), num=500)

    weights = np.array([minimizeVolatility(returns - r) for r in target_returns])
    returns = np.array([np.sum(weights[i]*returns.mean(axis=1)) for i in range(len(target_returns))])
    variances = np.array([np.dot(weights[i], np.dot(np.cov(returns.T), weights[i])) for i in range(len(target_returns))])
    return returns, variances, weights

# 绘制有效前沿
returns = np.array(returns)
frontier_returns, frontier_variances, frontier_weights = efficientFrontier(returns)
plt.plot(frontier_variances, frontier_returns, 'g-', label='Efficient Frontier')
# plt.scatter(np.sqrt(np.diagonal(np.cov(returns))), returns.mean(axis=1), color='b', label='Assets')
plt.xlabel('Volatility (standard deviation)')
plt.ylabel('Expected Returns')

plt.show()
