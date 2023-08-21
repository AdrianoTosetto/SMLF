import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from api.layers.activation.relu import ReLU
from api.layers.activation.sigmoid import Sigmoid
from api.layers.dense import Dense
from api.layers.dropout import Dropout
from api.losses.binary_cross_entropy import BinaryCrossEntropy
from api.metrics.accuracy.binary_accuracy import BinaryAccuracy
from api.metrics.probabilistic.binary_cross_entropy import BinaryCrossEntropy as BinaryCrossEntropyMetric
from api.optimizers.stochastic_gradient_descent import StochasticGradientDescent
from api.sequential.model import SequentialModel


sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))










dataframe = read_csv("/home/aot/Desktop/SMLF/examples/regularization/sonar.csv", header=None)
dataset = dataframe.values

X = dataset[:,0:60].astype(float)
Y = dataset[:,60]


encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y).reshape((208, 1))

X, Y = shuffle(X, Y)
print(X[0:10], Y[0:10])

X_train = X[0:180,:]
X_test = X[180:260,:]

Y_train = Y[0:180,:]
Y_test = Y[180:260,:]

model1 = SequentialModel(
    layers=[
        Dense(ninputs=60, noutputs=40, activation='relu'),
        Dropout(dropout_rate=.3, units=40),
        Dense(ninputs=40, noutputs=20, activation='sigmoid'),
        Dropout(dropout_rate=.3, units=20),
        Dense(ninputs=20, noutputs=10, activation='relu'),
        Dropout(dropout_rate=0.2,units=10),
        Dense(ninputs=10, noutputs=1, activation='sigmoid'),
        BinaryCrossEntropy(ninputs=1),
        
    ],
    optimizer=StochasticGradientDescent(),
    metrics=[BinaryAccuracy()]
)

model1.fit(X_train, Y_train, batch_size=90, epochs=10000)

model1.evaluate(X_test, Y_test)

model2 = SequentialModel(
    layers=[
        Dense(ninputs=60, noutputs=40, activation='relu'),
        Dense(ninputs=40, noutputs=20, activation='sigmoid'),
        Dense(ninputs=20, noutputs=10, activation='relu'),
        Dense(ninputs=10, noutputs=1, activation='sigmoid'),
        BinaryCrossEntropy(ninputs=1),
    ],
    optimizer=StochasticGradientDescent(),
    metrics=[BinaryAccuracy()]
)

model2.fit(X_train, Y_train, batch_size=60, epochs=10000)

model2.evaluate(X_test, Y_test)

print(model1.metrics[0].result())
print(model2.metrics[0].result())

# import tensorflow as tf

# m = tf.keras.metrics.BinaryCrossentropy()
# m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
# print(m.result().numpy())

# m = BinaryAccuracy()
# m.update_state(np.array([1, 1, 0, 0]).reshape(4,1), np.array([0.6, 0.5, 0.4, .6]).reshape(4,1))
# print(m.result())
