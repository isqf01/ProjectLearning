from SingleGPU.Mnist import MNIST
from SingleGPU.train import train


# 创建模型
model = MNIST()
# 启动训练过程
train(model)