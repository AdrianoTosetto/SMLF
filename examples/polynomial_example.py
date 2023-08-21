import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from api.layers.activation.relu import ReLU
from api.layers.activation.sigmoid import Sigmoid
from api.layers.dense import Dense
from api.layers.dense_l2_regularized import DenseRegularizedDecorator
from api.layers.inv_dropout import Dropout
from api.layers.softmax_categorical_cross_entropy_layer import SoftmaxCategoricalCrossEntropyLayer
from api.losses.regression.mean_squared_loss import MeanSquareError
from api.metrics.accuracy.categorical_accuracy import CategoricalAccuracy
from api.optimizers.adagrad_optimizer import AdagradOptmizer
from api.optimizers.stochastic_gradient_descent import StochasticGradientDescent
from api.sequential.model import SequentialModel






def f(x):
    return np.power(x, 7) - np.power(x, 4)  + np.power(x, 3) - 2.5*np.power(x, 2) + 1.5 + 2*np.power(np.random.random(size=(len(x), 1)), 2)

def plot(x1, y1, x2, y2):
    plt.title("Line graph")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.plot(x1, y1, 'b', x2, y2, 'r')
    plt.show()

dataset = np.arange(-1, 1.5, 0.2)
samples = dataset.shape[0]
dataset = dataset.reshape(samples, 1)

targets = f(dataset).reshape(samples, 1)

print(len(dataset))

model = SequentialModel(layers=[
    DenseRegularizedDecorator(ninputs=1, noutputs=20, activation=Sigmoid(units=20), weight_reg=0.00, bias_reg=0.00),
    DenseRegularizedDecorator(ninputs=20, noutputs=15, activation=Sigmoid(units=15), weight_reg=0.00, bias_reg=0.00),
    Dense(ninputs=15, noutputs=5, activation=Sigmoid(units=5)),
    Dense(ninputs=5, noutputs=1),
    MeanSquareError(units=1),
], optimizer= StochasticGradientDescent(learning_rate=0.01),
    metrics=[CategoricalAccuracy(),])

model.fit(dataset, targets, epochs=50000, batch_size=100)

plot(dataset, targets, dataset, model.predict(dataset))
