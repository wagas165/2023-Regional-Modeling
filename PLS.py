import numpy as np
from sklearn.cross_decomposition import PLSRegression

# 生成模拟数据
X = np.random.rand(100, 9) # 100个样本，每个样本有9个自变量
Y = np.random.rand(100, 3) # 100个样本，每个样本有3个因变量

# PLS回归
pls = PLSRegression(n_components=3) # 设置潜在变量的数量
pls.fit(X, Y) # 拟合PLS模型

# 预测
Y_pred = pls.predict(X) # 使用PLS模型进行预测
print("Predicted Y values:")
print(Y_pred)
print(pls.)
