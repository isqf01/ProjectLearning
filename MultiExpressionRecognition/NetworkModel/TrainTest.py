# 创建模型
import paddle

from NetworkModel.Train import train_pm
from NetworkModel.VGGNetwork import VGG, loss_fct, EPOCH_NUM

model = VGG()
# learning_rate为学习率，用于参数更新的计算。momentum为动量因子。
opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())

# 启动训练过程
train_pm(model, opt, loss_fct, EPOCH_NUM)
