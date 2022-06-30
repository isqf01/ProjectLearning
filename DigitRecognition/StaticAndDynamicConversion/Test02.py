import numpy as np
import paddle
import paddle.nn.functional as F
# 确保从paddle.vision.datasets.MNIST中加载的图像数据是np.ndarray类型
from StaticAndDynamicConversion.Train import norm_img

paddle.vision.set_image_backend('cv2')

# 读取mnist测试数据，获取第一个数据
mnist_test = paddle.vision.datasets.MNIST(mode='test')
test_image, label = mnist_test[0]
# 获取读取到的图像的数字标签
print("The label of readed image is : ", label)

# 将测试图像数据转换为tensor，并reshape为[1, 784]
test_image = paddle.reshape(paddle.to_tensor(test_image), [1, 784])
# 然后执行图像归一化
test_image = norm_img(test_image)
# 加载保存的模型
loaded_model = paddle.jit.load("./inference/mnist")
# 利用加载的模型执行预测
preds = loaded_model(test_image)
pred_label = paddle.argmax(preds)
# 打印预测结果
print("The predicted label is : ", pred_label.numpy())