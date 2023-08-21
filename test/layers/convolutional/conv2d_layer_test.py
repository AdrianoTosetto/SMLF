import math
import sys
import unittest

import numpy as np
from matplotlib import pyplot as plt

from api.conv2d_layer import Conv2DLayer


sys.path.append('../../..')


class TestReLU(unittest.TestCase):
    
    '''    
        Tests for checking if parameters shapes are being set correctly
    '''
    def test_output_shape_w_valid_padding(self):
        input_size = (10, 10, 3)
        padding_mode = 'valid'
        layer = Conv2DLayer(input_size=input_size, kernel_size=(3,3), depth=5, padding_mode=padding_mode)
        self.assertEqual(layer.output_shape, (5, 8, 8))

    def test_output_shape_w_same_padding(self):
        input_size = (3, 10, 10)
        padding_mode = 'same'
        nkernels = 5
        layer = Conv2DLayer(input_size=input_size, kernel_size=(3,3), depth=nkernels, padding_mode=padding_mode)
        self.assertEqual(layer.output_shape, (5, 10, 10))


    def test_kernel_and_biases_shape(self):
        input_size = (3, 10, 10)
        padding_mode = 'same'
        nkernels = 5
        layer = Conv2DLayer(input_size=input_size, kernel_size=(3,3), depth=nkernels, padding_mode=padding_mode)
        self.assertEqual(layer.kernel_size, (3, 3, 3))
        self.assertEqual(len(layer.kernels), nkernels)

        # self.assertEqual(layer.bias_shape, (10, 10))
        # self.assertEqual(len(layer.biases), nkernels)

    def test_single_volume_convolution_id_kernel(self):
        volume = np.array(range(0, 5*5*5)).reshape((5,5,5))
        '''
            pixel-wise sum across channels
        '''
        tmp = np.array(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ]
            )

        sum_across_channels_kernel = np.stack([tmp for _ in range(5)], axis=0)
        input_size = (5,5,5)
        padding_mode = 'valid'
        layer = Conv2DLayer(input_size=input_size, kernel_size=(3,3), depth=1, padding_mode=padding_mode)
        layer.kernels = [sum_across_channels_kernel]
        convolved = layer.convolve_single_volume(volume)
        target = np.sum(volume, axis=0, keepdims=True)
        self.assertTrue(np.allclose(convolved, target))


if __name__ == '__main__':
    unittest.main()
