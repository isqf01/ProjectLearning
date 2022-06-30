
from LossFunction.SimpleNet import MNIST
from LossFunction.Train import train


# 创建模型
model = MNIST()
# 启动训练过程
train(model)