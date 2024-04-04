import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

df3 = pd.read_excel('/Users/zhangyichi/Desktop/pythonProject/2023华东杯/对数收益率综合.xlsx',sheet_name='Sheet3')
df1=pd.read_excel('/Users/zhangyichi/Desktop/pythonProject/2023华东杯/对数收益率综合.xlsx',sheet_name='Sheet1')
df2=pd.read_excel('/Users/zhangyichi/Desktop/pythonProject/2023华东杯/对数收益率综合.xlsx',sheet_name='Sheet2')
df1=df1.iloc[:,1:]
df2=df2.iloc[:,1:]
df3=df3.iloc[:,1:]
df=pd.concat([df1,df2],axis=0)

# 对数据进行归一化
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
scaled_test_data = scaler.transform(df3)  # 注意我们在验证集上只使用transform，不使用fit

# 转化为torch tensor
tensor_data = torch.tensor(scaled_data, dtype=torch.float32)
tensor_test_data = torch.tensor(scaled_test_data, dtype=torch.float32)

# 定义一个批次大小为256的DataLoader
data_loader = DataLoader(tensor_data, batch_size=64, shuffle=True)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(9, 6),
            nn.ReLU(),
            nn.Linear(6, 3),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 6),
            nn.ReLU(),
            nn.Linear(6, 9),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 初始化模型和优化器
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.04)

# 创建一个空列表来存储每个周期的损失值
losses = []

# 训练自编码器
for epoch in range(500):
    for batch in data_loader:  # 这里使用了data_loader
        output = model(batch)
        loss = criterion(output, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 在每个周期结束时，将损失值添加到列表中
    losses.append(loss.item())

with torch.no_grad():
    output_test = model(tensor_test_data)
    loss_test = criterion(output_test, tensor_test_data)
print(f'Test loss={loss_test.item()}')

# 提取特征
encoded_data = model.encoder(tensor_data).detach().numpy()

# 创建一个3D子图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 将每个特征赋值给一个变量
x = encoded_data[:, 0]
y = encoded_data[:, 1]
z = encoded_data[:, 2]

# 创建一个3D散点图
ax.scatter(x, y, z)

# 设置轴标签
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')

# 显示图像
plt.show()

# 创建一个新的图形来绘制损失曲线
plt.figure()
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()