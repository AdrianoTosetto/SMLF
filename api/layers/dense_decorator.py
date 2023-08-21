import numpy as np

from api.layers.dense import Dense


class DenseDecorator(Dense):
    def __init__(self, decoratee: Dense, w_regularizer, b_regularizer):
        self.decoratee = decoratee
        self.w_regularizer = w_regularizer
        self.b_regularizer = b_regularizer

    def forward(self, batch_input: np.ndarray) -> None:
        self.decoratee.forward(batch_input)

    def backward(self, next_layer_backward: np.ndarray) -> None:
        self.decoratee.backward(next_layer_backward)
