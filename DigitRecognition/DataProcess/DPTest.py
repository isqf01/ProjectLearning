
# 创建模型
from DataProcess.NetworkStructure import MNIST, train

model = MNIST()
# 启动训练过程
train(model)