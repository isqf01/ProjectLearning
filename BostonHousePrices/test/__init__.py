# 获取数据
import numpy as np
from matplotlib import pyplot as plt

from loadData.load_data import load_data
from network.Network import Network

##########################################
# 测试load_data()函数
training_data, test_data = load_data()
x = training_data[:, :-1]
y = training_data[:, -1:]

# 查看数据
print(x[0])
print(y[0])
##########################################
print('##########这是个分割线###########')

##########################################
# 测试Network类中的forward()函数

net = Network(13)
x1 = x[0]
y1 = y[0]
z = net.forward(x1)
print(z)
###########################################
print('##########这是个分割线###########')

# 测试Network类中的loss()函数
net = Network(13)
# 此处可以一次性计算多个样本的预测值和损失函数
x1 = x[0:3]
y1 = y[0:3]
z = net.forward(x1)
print('predict: ', z)
loss = net.loss(z, y1)
print('loss:', loss)

###########################################
print('##########这是个分割线###########')

# 测试Network类中的gradient()函数
# 调用上面定义的gradient函数，计算梯度
# 初始化网络
net = Network(13)
# 设置[w5, w9] = [-100., -100.]
net.w[5] = -100.0
net.w[9] = -100.0

z = net.forward(x)
loss = net.loss(z, y)
gradient_w, gradient_b = net.gradient(x, y)
gradient_w5 = gradient_w[5][0]
gradient_w9 = gradient_w[9][0]
print('point {}, loss {}'.format([net.w[5][0], net.w[9][0]], loss))
print('gradient {}'.format([gradient_w5, gradient_w9]))

###########################################
# print('##########这是个分割线###########')
#
# # 获取数据
# train_data, test_data = load_data()
# x = train_data[:, :-1]
# y = train_data[:, -1:]
# # 创建网络
# net = Network(13)
# num_iterations=1000
# # 启动训练
# losses = net.train(x,y, iterations=num_iterations, eta=0.01)
#
# # 画出损失函数的变化趋势
# plot_x = np.arange(num_iterations)
# plot_y = np.array(losses)
# plt.plot(plot_x, plot_y)
# plt.show()

###########################################
print('##########这是个分割线###########')

# 获取数据
train_data, test_data = load_data()

# 创建网络
net = Network(13)
# 启动训练
losses = net.train(train_data, num_epochs=50, batch_size=100, eta=0.1)

# 画出损失函数的变化趋势
plot_x = np.arange(len(losses))
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()

# Numpy提供了save接口，可直接将模型权重数组保存为.npy格式的文件
np.save('../data/w.npy', net.w)
np.save('../data/b.npy', net.b)


###########################################
print('##########这是个分割线###########')
