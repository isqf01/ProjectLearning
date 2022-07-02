# 查看数据形状
from LoadData.load_data import data_loader

DATADIR = '../data/train'
train_loader = data_loader(DATADIR,batch_size=20, mode='train')
data_reader = train_loader()
data = next(data_reader) #返回迭代器的下一个项目给data
# 输出表示： 图像数据（batchsize，通道数，224*224）标签（batchsize，标签维度）
print("train mode's shape:")
print("data[0].shape = %s, data[1].shape = %s" %(data[0].shape, data[1].shape))

eval_loader = data_loader(DATADIR,batch_size=20, mode='eval')
data_reader = eval_loader()
data = next(data_reader)
# 输出表示： 图像数据（batchsize，通道数，224*224）标签（batchsize，标签维度）
print("eval mode's shape:")
print("data[0].shape = %s, data[1].shape = %s" %(data[0].shape, data[1].shape))
