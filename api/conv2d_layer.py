from typing import Union

import numpy as np
from scipy import signal

from api.common_layer import CommonLayer


def _calculate_output_shape(input_shape: tuple, kernel_size: tuple, depth: int, padding_mode: str = 'valid'):
    if (padding_mode == 'valid'):
        width = input_shape[0] - kernel_size[0] + 1
        height = input_shape[1] - kernel_size[1] + 1

        return (depth, width, height)

    if (padding_mode == 'same'):
        return (depth, input_shape[1], input_shape[2])

class Conv2DLayer(CommonLayer):
    
    '''
        self.input_size (depth, width, height)
        kernel_shape: (depth, width, height)
    '''
    
    def __init__(self, input_size: tuple, kernel_size: tuple | int, depth: int, padding_mode: str = 'valid'):
        self.input_size  = input_size
        self.depth = depth
        self.kernel_size = self._validate_kernel_size(kernel_size)
        self.output_shape = _calculate_output_shape(input_shape=input_size, kernel_size=kernel_size, depth=depth, padding_mode=padding_mode)
        self.bias_shape = (self.output_shape[0], self.output_shape[1])
        self._init_kernels()
        self._init_biases()

    def _init_kernels(self):
        self.kernels = []
        kernel_depth = self.input_size[0]
        kernel_width = self.kernel_size[1]
        kernel_height = self.kernel_size[2]

        for _ in range(0, self.depth):
            kernel = np.random.random((kernel_depth, kernel_width, kernel_height))
            self.kernels.append(kernel)

    def _init_biases(self):
        self.biases = []

        for _ in range(0, self.depth):
            biases_kernel = np.random.random(self.bias_shape)
            self.biases.append(biases_kernel)

    def _validate_kernel_size(self, kernel_size: tuple | int):
        if isinstance(kernel_size, tuple):
            if (len(kernel_size) != 2):
                raise Exception('Invalid kernel shape')
            return (self.input_size[0], kernel_size[0], kernel_size[1])

        if isinstance(kernel_size, int):
            return np.shape((kernel_size, kernel_size))

        kernel_size_type = type(kernel_size)
        raise Exception('Invalid kernel object, expected tuple | int but got type {type}'.format(type=kernel_size_type))

    '''
        (d, w, h)
    '''
    def convolve_single_volume(self, volume: np.ndarray):
        if len(volume.shape) != 3:
            raise Exception('Invalid volume shape')

        output_depth = len(self.kernels)

        output = np.zeros((output_depth, volume.shape[1], volume.shape[2]))

        for kernel, idx in zip(self.kernels, range(len(self.kernel_size))):            
            convolved = np.zeros((volume.shape[1], volume.shape[2]))
            depth = kernel.shape[0]

            for d in range(depth):
                convolved += signal.convolve2d(volume[d], kernel[d], mode='same')
            output[idx, ::] = convolved

        return output

    def forward(self, batch_input: np.ndarray):
        '''
            (m, width, height, channels)
        '''
        if (len(batch_input.shape) == 4):
            batch_size = batch_input.shape[0]

            output = np.zeros_like((batch_size) + self.output_shape)

            for idx in range(0, batch_size):
                single = batch_input[idx]
