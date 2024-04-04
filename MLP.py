import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 生成模拟数据
np.random.seed(0)
n_samples = 1000
n_features = 9
X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples)

# 划分训练集和测试集
train_ratio = 0.8
n_train = int(n_samples * train_ratio)
X_train, X_test = torch.tensor(X[:n_train], dtype=torch.float32), torch.tensor(X[n_train:], dtype=torch.float32)
y_train, y_test = torch.tensor(y[:n_train], dtype=torch.float32), torch.tensor(y[n_train:], dtype=torch.float32)

# 定义多层感知器模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建多层感知器模型
model = MLP(n_features, 64, 32)

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(50):
    optimizer.zero_grad()
    y_pred = model(X_train).squeeze()
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()

# 在测试集上评估模型性能
y_test_pred = model(X_test).squeeze()
mse_test = loss_fn(y_test_pred, y_test).item()
print(f"Test set MSE: {mse_test:.4f}")

# 预测
y_pred = model(X_test)
print(f"Predictions: {y_pred.squeeze().detach().numpy()}")