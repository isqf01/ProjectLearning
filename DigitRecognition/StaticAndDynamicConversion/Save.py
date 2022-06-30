import paddle

# save inference model
from paddle.static import InputSpec
# 加载训练好的模型参数
from StaticAndDynamicConversion.Test import model

state_dict = paddle.load("./mnist.pdparams")
# 将训练好的参数读取到网络中
model.set_state_dict(state_dict)
# 设置模型为评估模式
model.eval()

# 保存inference模型
paddle.jit.save(
    layer=model,
    path="inference/mnist",
    input_spec=[InputSpec(shape=[None, 784], dtype='float32')])

print("==>Inference model saved in inference/mnist.")