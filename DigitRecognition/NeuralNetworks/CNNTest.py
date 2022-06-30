# 创建模型
from DataProcess.NetworkStructure import train
from NeuralNetworks.CNN import MNIST

model = MNIST()
# 启动训练过程
train(model)