import paddle

from StaticAndDynamicConversion.Network import MNIST
from StaticAndDynamicConversion.Train import train

model = MNIST()

train(model)

paddle.save(model.state_dict(), './mnist.pdparams')
print("==>Trained model saved in ./mnist.pdparams")