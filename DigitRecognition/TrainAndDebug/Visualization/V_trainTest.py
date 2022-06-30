import matplotlib.pyplot as plt

from TrainAndDebug.Checking.Network import MNIST
from TrainAndDebug.Visualization.V_train import train

model = MNIST()
iters, losses = train(model)

#画出训练过程中Loss的变化曲线
plt.figure()
plt.title("train loss", fontsize=24)
plt.xlabel("iter", fontsize=14)
plt.ylabel("loss", fontsize=14)
plt.plot(iters, losses,color='red',label='train loss')
plt.grid()
plt.show()