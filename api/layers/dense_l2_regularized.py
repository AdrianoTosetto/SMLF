import numpy as np

from api.common_layer import CommonLayer
from api.internal.smlf_component import SMLFComponent
from api.layers.dense import Dense
from api.layers.dense_decorator import DenseDecorator


'''
    this is a slightly different use of decorator pattern for applying L2 regularization where the 
    decoratee is created within the decorator __init__ method
'''

class DenseL2Regularized(SMLFComponent, CommonLayer):
    def __init__(self, ninputs, noutputs, name: str = 'Dense L2', init_algorithm='uniform', activation: str | CommonLayer=None,
                 weight_reg = 1.0,
                 bias_reg = 1.0,
                 ):

        SMLFComponent.__init__(self)
        CommonLayer.__init__(self, ninputs, noutputs)        

        self.w_regularizer = weight_reg
        self.b_regularizer = bias_reg
        self.decoratee = Dense(ninputs=ninputs, noutputs=noutputs, init_algorithm=init_algorithm)

    def regularization_term_weights(self,):
        return self.w_regularizer *  np.sum(np.power(self.decoratee.weights, 2))

    def regularization_term_biases(self,):
        return self.b_regularizer * np.sum(np.power(self.decoratee.biases, 2))

    def forward(self, batch_input: np.ndarray) -> None:
        self.decoratee.forward(batch_input)

    def backward(self, next_layer_backward: np.ndarray) -> None:
        self.decoratee.backward(next_layer_backward)
        
        self.decoratee.derivatives_wrt_weights += 2 * self.w_regularizer * self.decoratee.weights
        self.decoratee.derivatives_wrt_biases += 2 * self.b_regularizer * self.decoratee.biases
