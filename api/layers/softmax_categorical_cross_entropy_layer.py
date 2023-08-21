import numpy as np
import tensorflow as tf

from api import common_layer
from api.losses import common_loss


class SoftmaxCategoricalCrossEntropyLayer(common_layer.CommonLayer):
    def __init__(self, units) -> None:
        super().__init__(units, units)
        self.units = units

    def simple_forward(self, batch_input):
        batch_size = batch_input.shape[0]
        batch_input_normalized = batch_input - np.max(batch_input, axis=1, keepdims=True)
        exponentials = np.exp(batch_input_normalized)

        # print('sum = ' + str(np.sum(exponentials)))
        # exponentials_sums = np.sum(exponentials, axis=1, keepdims=True)
        self.output = exponentials / np.sum(exponentials, axis=1, keepdims=True)         

    def forward(self, batch_input: np.ndarray, batch_targets: np.ndarray) -> None:
        batch_size = batch_input.shape[0]
        batch_input_normalized = batch_input - np.max(batch_input, axis=1, keepdims=True)
        exponentials = np.exp(batch_input_normalized)

        # print('sum = ' + str(np.sum(exponentials)))
        # exponentials_sums = np.sum(exponentials, axis=1, keepdims=True)
        self.output = exponentials / np.sum(exponentials, axis=1, keepdims=True) 
        
        tmp = -np.log(np.clip(self.output, 1e-7, 1 - 1e-7)) * batch_targets
        self.loss = np.sum(tmp) / batch_size

    def backward(self, targets: np.ndarray) -> None:
        self.derivatives_wrt_inputs = (self.output - targets) / targets.shape[0]
