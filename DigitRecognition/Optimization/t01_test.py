# 创建模型
from LossFunction.SimpleNet import MNIST
from LossFunction.Train import train

model = MNIST()
# 启动训练过程
train(model)