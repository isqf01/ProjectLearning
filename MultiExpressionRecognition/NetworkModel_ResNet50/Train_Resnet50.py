import paddle
import os
import random
import numpy as np

from LoadData.load_data import data_loader

DATADIR = '../data/train'
DATADIR2 = '../data/val'
DATADIR3 = '../data/test'


def train_pm(model, optimizer, loss_fct,
             EPOCH_NUM):  # optimizer表示优化器 loss_fict 为损失函数 EPOCH_NUM为迭代次数 这些之前训练vgg网络时已经定义过，无需再次定义
    # 开启0号GPU训练
    use_gpu = True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

    print('start training ... ')
    model.train()
    # 定义数据读取器，训练数据读取器和验证数据读取器
    train_loader = data_loader(DATADIR, batch_size=20, mode='train')
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            x_data, y_data = data

            # 将图片和标签都转化为tensor型
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)

            # 运行模型前向计算，得到预测值
            logits = model(img)
            # 计算输入input和标签label间的交叉熵损失
            avg_loss = loss_fct(logits, label)

            if batch_id % 200 == 0:
                print("epoch: {}, batch_id: {}, loss is: {:.4f}".format(epoch, batch_id, float(avg_loss.numpy())))

            # 反向传播，更新权重，清除梯度
            avg_loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        # 保存模型
        paddle.save(model.state_dict(), 'resnet50.pdparams')
        paddle.save(optimizer.state_dict(), 'resnet50.pdopt')
