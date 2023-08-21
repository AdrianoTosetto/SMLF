import math
import sys
import unittest

import numpy as np
from matplotlib import pyplot as plt

from api.layers.activation.relu import ReLU


sys.path.append('../../..')


class TestReLU(unittest.TestCase):
    
    def test_shapes_on_forward(self):
        layer = ReLU(5)
        batch = np.ndarray((2, 5), buffer=np.array([
            [-1., .0, .1, 1, -1.],
            [-1., .0, .1, 1, -1.],
        ]))
        
        layer.forward(batch)
        self.assertEqual(layer.output.shape, (2, 5))

    def test_output_values_on_forward(self):
        layer = ReLU(5)
        batch = np.ndarray((1, 5), buffer=np.array([
            [-1., 0, .1, .1, .1]
        ]))
    
        layer.forward(batch)
        np.testing.assert_array_equal(layer.output, np.array([[.0, .0, .1, .1, .1]]))

    def test_full_batch_forward(self):
        ninputs = 30
        batch_size = 150
        layer = ReLU(1000)
        batch = np.random.uniform(-1, 1, (batch_size, ninputs))
        derivatives_previous_layer = np.ones((batch_size, ninputs))

        layer.forward(batch)
        layer.backward(derivatives_previous_layer)

        value1 = np.sum(derivatives_previous_layer, where=np.greater(layer.output, 0))
        value2 = np.sum(layer.derivatives_wrt_inputs)

        self.assertEqual(value1, value2)
        self.assertEqual(layer.output.shape, (batch_size, ninputs))
        self.assertEqual(layer.derivatives_wrt_inputs.shape, (batch_size, ninputs))


    def test_derivatives_on_backprop(self):
        layer = ReLU(5)
        batch = np.ndarray((1, 5), buffer=np.array([
            [-1., 0, .1, .1, .1]
        ]))

        derivatives_previous_layer = np.ndarray((1, 5), buffer=np.array([
            [1., 1., 1., 1., 1.]
        ]))

        layer.forward(batch)
        layer.backward(derivatives_previous_layer)

        np.testing.assert_array_equal(layer.derivatives_wrt_inputs, np.array([[0., 0., 1, 1, 1]]))


if __name__ == '__main__':
    unittest.main()
