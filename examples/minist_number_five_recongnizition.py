import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from api.batch_dense_normalization import BatchDenseNormalization
from api.layers.activation.relu import ReLU
from api.layers.activation.sigmoid import Sigmoid
from api.layers.activation.softmax import Softmax
from api.layers.dense import Dense
from api.layers.inv_dropout import InvDropout
from api.layers.softmax_categorical_cross_entropy_layer import SoftmaxCategoricalCrossEntropyLayer
from api.losses.regression.mean_squared_loss import MeanSquareError
from api.optimizers.adagrad_optimizer import AdagradOptmizer
from api.optimizers.stochastic_gradient_descent import StochasticGradientDescent
from api.sequential.model import SequentialModel


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

input_size = 28 * 28
train_size = len(x_train)
test_size = len(x_test)
num_classes = 10

''''
    reshape x_train, x_test from 28x28 matrix to 784-element vector
    work on data, get y_train, y_test to be one-hot encoded
    normalize to range 0, 1
'''

x_train = np.reshape(x_train, (train_size, input_size))
x_test = np.reshape(x_test, (test_size, input_size))

x_train = x_train / 255.0
x_test = x_test / 255.0


y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

init_algorithm = 'random'

model = SequentialModel(layers=[
    Dense(ninputs=784, noutputs=256, init_algorithm=init_algorithm),
    BatchDenseNormalization(units=256),
    ReLU(units=256),
    Dense(ninputs=256, noutputs=128, init_algorithm=init_algorithm),
    BatchDenseNormalization(units=128),
    ReLU(units=128),
    Dense(ninputs=128, noutputs=10, init_algorithm=init_algorithm),
    BatchDenseNormalization(units=10),
    Softmax(units=10),
], optimizer=AdagradOptmizer(learning_rate=1.0), loss='categorical_cross_entropy')

small_test_x = x_test[0:9,:]
small_test_y = y_test[0:9,:]

print(model.predict(small_test_x).shape)

predictions = np.argmax(model.predict(small_test_x), axis=1)
correct_classes = np.argmax(small_test_y, axis=1)

print(predictions)
print(correct_classes)

fig, ax = plt.subplots(3, 3)
fig.suptitle('Before training')

for image, i in zip(small_test_x, range(len(small_test_x))) :
    ax[i // 3, i % 3].imshow(image.reshape(28, 28), cmap='gray')
    ax[i // 3, i % 3].set_title('Predicted: {}. Correct: {}'.format(predictions[i], correct_classes[i]))
    


print(predictions)
print(correct_classes)

model.fit(x_train, y_train, epochs=300, batch_size=400)

predictions = np.argmax(model.predict(small_test_x), axis=1)
correct_classes = np.argmax(small_test_y, axis=1)

fig, ax = plt.subplots(3, 3)
fig.suptitle('After training')

for image, i in zip(small_test_x, range(len(small_test_x))) :
    ax[i // 3, i % 3].imshow(image.reshape(28, 28), cmap='gray')
    ax[i // 3, i % 3].set_title('Predicted: {}. Correct: {}'.format(predictions[i], correct_classes[i]))
    
print(predictions)
print(correct_classes)

plt.show()

