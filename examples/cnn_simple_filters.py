import os
import sys

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy import signal

from api.conv2d_layer import Conv2DLayer


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



matrix = np.array(range(1, 26)).reshape((5,5))
kernel = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

print(matrix)
# print(kernel.shape)

convolved = signal.convolve2d(matrix, kernel, mode='same')
print(convolved)

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# image = x_train[0].reshape(1, 28, 28)

# identity_filter = np.array([
#     [0, 0, 0],
#     [0, 1, 0],
#     [0, 0, 0],
# ]).reshape((1, 3, 3))

# edge_detection_filter = np.array([
#     [-1, -1, -1],
#     [-1, 8, -1],
#     [-1, -1, -1],
# ]).reshape((1, 3, 3))

# gaussian_blur = np.array([
#     [1, 2, 1],
#     [2, 4, 2],
#     [1, 2, 1],
# ]).reshape((1, 3, 3)) * (1 / 16)

# filters = [identity_filter, edge_detection_filter, gaussian_blur]
# print(gaussian_blur.shape)

# layer = Conv2DLayer(depth=3, input_size=(1, 28, 28), kernel_size=(3,3), padding_mode='same')
# layer.kernels = filters
# output = layer.convolve_single_volume(image)

# fig, ax = plt.subplots(2, 2)
# fig.suptitle('Filters')

# ax[0, 0].imshow(image.reshape(28, 28), cmap='gray')
# ax[0, 1].imshow(output[0], cmap='gray')
# ax[1, 0].imshow(output[1], cmap='gray')
# ax[1, 1].imshow(output[2], cmap='gray')

# plt.show()
