import os
import sys

import numpy as np
from sklearn.utils import shuffle

from api.batch_dense_normalization import BatchDenseNormalization
from api.layers.activation.relu import ReLU
from api.layers.activation.sigmoid import Sigmoid
from api.layers.activation.softmax import Softmax
from api.layers.dense import Dense
from api.losses.binary_cross_entropy import BinaryCrossEntropy
from api.optimizers.adagrad_optimizer import AdagradOptmizer
from api.optimizers.stochastic_gradient_descent import StochasticGradientDescent
from api.sequential.model import SequentialModel


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))






samples = 4

rng = np.random.default_rng()
dataset = rng.uniform(0., 1., (samples, 2))
# dataset = np.array([
#     [0., 0.],
#     [0., 1.],
#     [1., 0.],
#     [1., 1.],
# ])

def generate_targets(input):
    if (input[0] < .5 and input[1] < .5):
        return np.array([0.])
    if (input[0] >= .5 and input[1] >= .5):
        return np.array([0.])

    return np.array([1.])

targets = np.apply_along_axis(generate_targets, axis=1, arr=dataset).reshape(samples, 1)

print(dataset)
print(targets)
non_zeros = np.count_nonzero(targets)
zeros = samples - non_zeros

per_non_zeros = non_zeros / samples
per_zeros = zeros / samples

print(f'zero per: {per_zeros} ones per: {per_non_zeros}')

# the four different states of the XOR gate
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

# the four expected results in the same order
target_data = np.array([[0],[1],[1],[0]], "float32")

dataset_test = np.array([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])

targets_test = np.array([
    [0.],
    [1.],
    [1.],
    [0.]
])

model = SequentialModel(
    layers=[
        Dense(ninputs=2, noutputs=8, init_algorithm='uniform', activation='relu'),
        Dense(ninputs=8, noutputs=1, init_algorithm='uniform'),
        Sigmoid(1)
    ],
    optimizer=StochasticGradientDescent(learning_rate=0.01), loss='binary_cross_entropy'
)

model.fit(training_data, target_data, epochs=10)
