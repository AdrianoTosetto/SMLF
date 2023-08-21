import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

from api.batch_dense_normalization import BatchDenseNormalization
from api.layers.activation.relu import ReLU
from api.layers.activation.sigmoid import Sigmoid
from api.layers.dense import Dense
from api.losses.regression.mean_squared_loss import MeanSquareError
from api.optimizers.adagrad_optimizer import AdagradOptmizer
from api.optimizers.stochastic_gradient_descent import StochasticGradientDescent
from api.sequential.model import SequentialModel






epochs = 10000

init_algorithm = 'random'


model = SequentialModel(
    layers=[
        Dense(ninputs=1, noutputs=8, init_algorithm=init_algorithm),
        BatchDenseNormalization(8),
        ReLU(units=8),
        Dense(ninputs=8, noutputs=5, init_algorithm=init_algorithm),
        # BatchDenseNormalization(10),
        ReLU(units=5),
        Dense(ninputs=5, noutputs=1, init_algorithm=init_algorithm),
    ],
    loss = MeanSquareError(1),
    optimizer=AdagradOptmizer(learning_rate=1.)
)

# def plot(x, y):
#     plt.title("Line graph")
#     plt.xlabel("X axis")
#     plt.ylabel("Y axis")
#     plt.plot(x, y, color ="red")
#     plt.show()

dataset = np.arange(0., 2*np.pi, 0.001)
samples = dataset.shape[0]
targets = (np.sin(dataset)).reshape(samples, 1)
dataset = dataset.reshape(samples, 1)

model.fit(dataset, targets, batch_size=len(dataset), epochs=10000)

targets_test = model.predict(dataset)

plt.plot(dataset, targets, 'b', dataset, targets_test, 'r--')
plt.show()
