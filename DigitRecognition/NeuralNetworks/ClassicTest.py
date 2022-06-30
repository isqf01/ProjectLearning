from DataProcess.NetworkStructure import train
from NeuralNetworks.Classic import MNIST

# 创建模型
model = MNIST()
# 启动训练过程
train(model)