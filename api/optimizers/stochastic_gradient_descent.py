import numpy as np

from api.batch_dense_normalization import BatchDenseNormalization
from api.internal.smlf_component import SMLFComponent
from api.layers.dense import Dense
from api.optimizers.optimizer_common import OptimizerCommon


class StochasticGradientDescent(SMLFComponent, OptimizerCommon):
    def __init__(self, learning_rate=0.001, momentum=0.0, name: str = 'sgd') -> None:
        OptimizerCommon.__init__(self, learning_rate)
        SMLFComponent.__init__(self)
        self._name = name
        self.momentum = momentum
        self.use_momentum = False

        if self.momentum > 0.0:
            self.use_momentum = True
        
    def update_weights(self, layer: Dense):
        if self.use_momentum:

            layer.momentum_weights = self.momentum * layer.momentum_weights - self.learning_rate * layer.derivatives_wrt_weights
            layer.momentum_biases = self.momentum * layer.momentum_biases - self.learning_rate * layer.derivatives_wrt_biases

            layer.weights += layer.momentum_weights
            layer.biases += layer.momentum_biases

        else:
            layer.weights += -self.learning_rate * layer.derivatives_wrt_weights
            layer.biases += -self.learning_rate * layer.derivatives_wrt_biases

    def update_params(self, layer: BatchDenseNormalization):
        layer.gamma = layer.gamma - self.learning_rate * layer.derivatives_wrt_gamma
        layer.beta = layer.beta - self.learning_rate * layer.derivatives_wrt_beta

    '''
        SMLF Base Component overrides
    '''

    def name(self):
        return self._name

    def component_name(self):
        return 'StochasticGradientDescent'

    def save_state():
        pass

    def to_json(self):
        json = dict({
            'name': self.name,
            'component_name': self.component_name(),
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
        })

        return json

    def to_string_json(self):
        pass

    @classmethod
    def from_json(cls, json_obj: dict):
        name = json_obj['name']
        learning_rate = float(json_obj['learning_rate'])
        json_momentum = json_obj['momentum']
        momentum = float(json_momentum) if json_momentum is not None else 0.0

        return cls(learning_rate=learning_rate, name=name, momentum=momentum)

    @classmethod
    def from_string_json(cls, json_str: str):
        pass
