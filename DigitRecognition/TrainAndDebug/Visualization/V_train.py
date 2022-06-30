# 引入matplotlib库
import matplotlib.pyplot as plt
import paddle
import paddle.nn.functional as F
from LossFunction.Dataset import train_loader


def train(model):
    model.train()
    opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    EPOCH_NUM = 10
    iter = 0
    iters = []
    losses = []
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            # 准备数据，变得更加简洁
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)
            # 前向计算的过程，同时拿到模型输出值和分类准确率
            predicts, acc = model(images, labels)
            # 计算损失，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)
            # 每训练了100批次的数据，打印下当前Loss的情况
            if batch_id % 100 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(),
                                                                            acc.numpy()))
                iters.append(iter)
                losses.append(avg_loss.numpy())
                iter = iter + 100
            # 后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()
    # 保存模型参数
    paddle.save(model.state_dict(), 'mnist.pdparams')
    return iters, losses
