import numpy as np
import tensorflow as tf

from api.common_layer import CommonLayer
from api.internal.smlf_component import SMLFComponent


ep = 1e-7

class BatchDenseNormalization(SMLFComponent, CommonLayer):

    def __init__(self, units: int):
        SMLFComponent.__init__(self)
        CommonLayer.__init__(self,ninputs=units, noutputs=units)
        self.gamma = np.ones((1, units))
        self.beta = np.zeros((1, units))
        self.units = units

    def forward(self, batch_input: np.ndarray, mode: str = 'train') -> None:
        self.batch_input = batch_input
        self.batch_features_mean = np.mean(batch_input, axis=0, keepdims=True)
        self.batch_features_variance = np.var(batch_input, axis=0, keepdims=True)
        self.samples = batch_input.shape[0]

        self.batch_input_norm = (batch_input - self.batch_features_mean) / np.sqrt((self.batch_features_variance + ep))
        self.output = self.batch_input_norm * self.gamma + self.beta

    def backward(self, backward_derivatives: np.ndarray) -> None:
        tmp =  (self.samples - 1 - (np.power(self.batch_input - self.batch_features_mean, 2) / (self.batch_features_variance + ep))) \
            / (self.samples * np.sqrt(self.batch_features_variance+ep))
        self.derivatives_wrt_inputs = (self.gamma * tmp) * backward_derivatives
        self.derivatives_wrt_gamma = np.sum(backward_derivatives * self.batch_input_norm, axis=0, keepdims=True)
        self.derivatives_wrt_beta = np.sum(backward_derivatives, axis=0, keepdims=True)

