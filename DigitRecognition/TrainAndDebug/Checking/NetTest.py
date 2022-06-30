from TrainAndDebug.Checking.NetTrain import train
from TrainAndDebug.Checking.Network import MNIST


# 创建模型
model = MNIST()
# 启动训练过程
train(model)

print("Model has been saved.")