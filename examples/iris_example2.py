import os
import sys

import numpy as np
import sklearn.utils
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize
from sklearn.utils import shuffle

from api.batch_dense_normalization import BatchDenseNormalization
from api.layers.activation.relu import ReLU
from api.layers.activation.sigmoid import Sigmoid
from api.layers.activation.softmax import Softmax
from api.layers.dense import Dense
from api.layers.softmax_categorical_cross_entropy_layer import SoftmaxCategoricalCrossEntropyLayer
from api.losses.regression.mean_squared_loss import MeanSquareError
from api.optimizers.adagrad_optimizer import AdagradOptmizer
from api.optimizers.stochastic_gradient_descent import StochasticGradientDescent
from api.sequential.model import SequentialModel


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

# One hot encoding
enc = OneHotEncoder()
Y = enc.fit_transform(y[:, np.newaxis]).toarray()

# Scale data to have mean 0 and variance 1 
# which is importance for convergence of the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X[0:10])
print(X_scaled[0:10])
# Split the data set into training and testing
supy = Y[50:]
supx = X_scaled[50:]
shuffle(supx, supy)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.5, random_state=2)

n_features = X.shape[1]
n_classes = Y.shape[1]

Y_train = Y_train[:, 0].reshape(-1, 1)
Y_test = Y_test[:, 0].reshape(-1, 1)

init_algorithm = 'uniform'

model = SequentialModel(
    layers=[
        Dense(ninputs=4, noutputs=3, init_algorithm=init_algorithm),
        # BatchDenseNormalization(units=3),
        ReLU(units=3),
        Dense(ninputs=3, noutputs=1, init_algorithm=init_algorithm),
        # BatchDenseNormalization(units=8),
        # ReLU(units=8),
        # Dense(ninputs=8, noutputs=1, init_algorithm=init_algorithm),
        # BatchDenseNormalization(units=3),
        Sigmoid(units=1),
    ],
    optimizer=AdagradOptmizer(learning_rate=1.),
    loss='binary_cross_entropy'
)

model.fit(X_train, Y_train, epochs=1000, batch_size=1)

# layer = SoftmaxCategoricalCrossEntropyLayer(ninputs=3)
# layer.forward(np.array([70, 10, 20]).reshape(1, 3), np.array([1., 0., 0.]).reshape(1, 3))
# print(layer.output)
# # # layer.output = np.array([0.7, 0.1, 0.2, 0.1, 0.5, 0.4, 0.02, 0.9, 0.08]).reshape(3, 3)
# # print(np.array([1., 0., 0., 0., 1, 0., 0., 1., 0.]).reshape(3, 3))
# # # layer.forward(np.array([5, 50, 3]).reshape(1, 3), np.array([0., 0., 1.]).reshape(1, 3),)
# # layer.backward(np.array([1., 0., 0., 0., 1., 0., 0., 1., 0.]).reshape(3, 3))
# # print(layer.derivatives_wrt_inputs)
