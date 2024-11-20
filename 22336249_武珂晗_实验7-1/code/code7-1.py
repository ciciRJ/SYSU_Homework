import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 符号函数
def sign(v):
    if v >= 0:
        return 1
    else:
        return -1

# def train(train_num, train_data, lr=0.5):  # 原始训练部分
#     w = [1., 1.]  # 初始化权重向量和偏置
#     b = -1
#     for i in range(train_num):
#         #x = random.choice(train_data)
#         for x in train_data:
#             x1, x2, y = x
#             if y * sign(x1 * w[0] + x2 * w[1] + b) <= 0:
#                 w[0] += lr * y * x1
#                 w[1] += lr * y * x2
#                 b += lr * y
#                 #print(w,b)
#     return w, b

#单层感知机
class Perceptron:
    #初始化感知器的权重、偏置和学习率
    def __init__(self, input_dim, lr=0.3):
        self.weights = np.random.randn(input_dim, 1)
        self.bias = np.random.randn(1)
        self.lr = lr    #学习率
        return

    #接收输入 x，计算加权和并通过激活函数得到预测结果
    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias
        return 1 / (1 + np.exp(-z))

    #损失函数，接收预测值 y_pred 和真实值y，计算二者之间的均方误差
    def loss(self, y_pred, y):
        return np.mean((y_pred - y) ** 2)

    #根据损失计算权重和偏置的梯度，更新权重和偏置
    def backprop(self, x, y):
        y_pred = self.forward(x)
        error = y_pred - y
        d_weights = np.dot(x.T, 2 * error * y_pred * (1 - y_pred)) / len(y)
        d_bias = np.sum(2 * error * y_pred * (1 - y_pred)) / len(y)
        self.weights -= self.lr * d_weights
        self.bias -= self.lr * d_bias
        return

# 画散点图
def plot_points(data, model):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    X = data.iloc[:, :-1]
    X0 = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    Y = data.iloc[:, -1]

    ax1.scatter(X.iloc[:, 0][Y == 0], X.iloc[:, 1][Y == 0], c='blue', label='Not Purchased')
    ax1.scatter(X.iloc[:, 0][Y == 1], X.iloc[:, 1][Y == 1], c='red', label='Purchased')
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Salary')
    ax1.legend()

    ax2.scatter(X0.iloc[:, 0][Y == 0], X0.iloc[:, 1][Y == 0], c='blue', label='Not Purchased')
    ax2.scatter(X0.iloc[:, 0][Y == 1], X0.iloc[:, 1][Y == 1], c='red', label='Purchased')
    w1, w2 = model.weights
    b = model.bias
    x_line =  np.array([X0.iloc[:,0].min(),X0.iloc[:,0].max()])
    y_line = (-w1 / w2) * x_line - (b / w2)
    ax2.plot(x_line, y_line, color='purple', label='Decision Boundary')
    ax2.set_title('Predictions')
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Salary')
    ax2.legend()

    plt.show()



if __name__ == '__main__':
    data = pd.read_csv('data.csv',delimiter=",",skiprows=0)
    # 归一化特征
    inputs = (data.iloc[:, :-1] - data.iloc[:, :-1].min()) / (data.iloc[:, :-1].max() - data.iloc[:, :-1].min())
    labels = data.iloc[:, -1]

    EPOCHS = 4000
    BATCH_SIZE = 32
    LOSS_LIST = []
    model = Perceptron(inputs.shape[1])

    for epoch in range(EPOCHS):
        for i in range(0, inputs.shape[0], BATCH_SIZE):
            x_batch = inputs.iloc[i:i + BATCH_SIZE].values
            y_batch = data.iloc[:, -1].iloc[i:i + BATCH_SIZE].values.reshape(-1, 1)
            y_pred = model.forward(x_batch)
            current_loss = model.loss(y_pred, y_batch)
            model.backprop(x_batch, y_batch)
        LOSS_LIST.append(current_loss)
        if (epoch + 1) % 500 == 0:
            print(f'Epoch: {epoch + 1:5d} Loss: {current_loss:.6f}')

    results = (model.forward(inputs) > 0.5).astype(np.int32)
    acc = (results == np.array(labels).reshape(-1, 1)).astype(np.int32).mean()
    print(f"The accuracy is: {acc * 100:.2f}%")

    plt.plot(range(1, EPOCHS + 1), LOSS_LIST, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.text(EPOCHS, LOSS_LIST[-1], f'Final Loss: {LOSS_LIST[-1]:.4f}', ha='right', va='baseline')
    plt.legend()

    plot_points(data, model)

