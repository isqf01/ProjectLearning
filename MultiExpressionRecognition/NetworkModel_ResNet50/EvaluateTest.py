import paddle
from NetworkModel.Evaluate import evaluation
from NetworkModel.VGGNetwork import loss_fct
from NetworkModel_ResNet50.ResNet50Test import model

# 开启0号GPU预估
use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
#model = paddle.vision.models.resnet50(pretrained=True, num_classes=3)
params_file_path = './resnet50.pdparams'
model_state_dict = paddle.load(params_file_path)
model.load_dict(model_state_dict)
# 调用验证
evaluation(model, loss_fct)
