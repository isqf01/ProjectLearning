import cv2
import matplotlib.pyplot as plt
import numpy as np
import paddle

from NetworkModel.VGGNetwork import VGG
# matplotlib inline
from PIL import Image


def load_image(img):
    # 将图片尺寸缩放道 224x224
    img = cv2.resize(img, (224, 224))
    # 读入的图像数据格式是[H, W, C]
    # 使用转置操作将其变成[C, H, W]
    img = np.transpose(img, (2, 0, 1))
    img = img.astype('float32')
    # 将数据范围调整到[-1.0, 1.0]之间
    img = img / 255.
    img = img * 2.0 - 1.0
    return img


model = VGG()
params_file_path = '../NetworkModel/vgg.pdparams'
# 选取test文件夹里一张表情图片 H开头表示happy
img_path = '../data/test/H00528.jpg'
img = cv2.imread(img_path)
plt.imshow(img)
plt.axis('off')
plt.show()

[height, width, pixels] = img.shape
frame = cv2.resize(img, (int(width / 3), int(height / 3)), interpolation=cv2.INTER_CUBIC)  # 缩小图像

param_dict = paddle.load(params_file_path)
model.load_dict(param_dict)
# 灌入数据
model.eval()
tensor_img = load_image(img)
tensor_img = np.expand_dims(tensor_img, 0)

results = model(paddle.to_tensor(tensor_img))
# 取概率最大的标签作为预测输出
lab = np.argsort(results.numpy())

print("这次预测图片名称是：%s" % (img_path[5:]))
if img_path[5] == 'N':
    true_lab = 'NORMAL'
elif img_path[5] == 'H':
    true_lab = 'HAPPY'
elif img_path[5] == 'R':
    true_lab = 'SUPRISE'
else:
    raise ('Not excepted file name')
print("这次图片属于%s表情" % (true_lab))
tap = lab[0][-1]
print("这次预测结果是：")
if tap == 0:
    print('NORMAL')
elif tap == 1:
    print('HAPPY')
elif tap == 2:
    print('SURPRISE')
else:
    raise ('Not excepted file name')
