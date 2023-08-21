import numpy as np
import tensorflow as tf

from api.dense import Dense
from relu_layer import ReLU
from softmax_categorical_cross_entropy_layer import SoftmaxCategoricalCrossEntropyLayer


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


'''
    model definition
'''

input_layer = Dense(ninputs=input_size, noutputs=128)
activation_input_layer = ReLU(ninputs=128)
second_layer = Dense(ninputs=128, noutputs=128)
activation_second_layer = ReLU(ninputs=128)
final_layer = Dense(ninputs=128, noutputs=10)
activation_final_layer = SoftmaxCategoricalCrossEntropyLayer(10)

epochs = 10001

x_train = x_train[0:1]
y_train = y_train[0:1]
# input_layer.forward(x_train)
# print(input_layer.output)

for epoch in range(1, epochs):
    # if (epoch % 100 == 0):
        # print(f'epoch: {epoch} loss = {activation_final_layer.loss}')
        # print(f'epoch: {epoch} ')
        # print(second_layer.output)
        # print(x_train)

    input_layer.forward(x_train)
    activation_input_layer.forward(input_layer.output)
    second_layer.forward(activation_input_layer.output)
    activation_second_layer.forward(second_layer.output)
    final_layer.forward(activation_second_layer.output)
    activation_final_layer.forward(final_layer.output, y_train)

