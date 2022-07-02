# 开启0号GPU预估
import paddle
from NetworkModel.Evaluate import evaluation
from NetworkModel.TrainTest import model
from NetworkModel.VGGNetwork import loss_fct

use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
# 加载模型参数
params_file_path = 'vgg.pdparams'
model_state_dict = paddle.load(params_file_path)
model.load_dict(model_state_dict)
# 调用验证
evaluation(model, loss_fct)
