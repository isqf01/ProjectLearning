import paddle

from NetworkModel.VGGNetwork import loss_fct, EPOCH_NUM
from NetworkModel_ResNet50.Train_Resnet50 import train_pm

# pretrained：表示是否加载在imagenet数据集上的预训练权重 num_classes由数据集的标签数决定
model = paddle.vision.models.resnet50(pretrained=True, num_classes=3)
# 选择momentum优化器
opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters(), weight_decay=0.001)
# 启动训练过程
train_pm(model, opt, loss_fct, EPOCH_NUM)
