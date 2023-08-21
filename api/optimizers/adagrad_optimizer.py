import numpy as np

from api.batch_dense_normalization import BatchDenseNormalization
from api.internal.smlf_component import SMLFComponent
from api.layers.dense import Dense
from api.optimizers.optimizer_common import OptimizerCommon


class AdagradOptmizer(SMLFComponent, OptimizerCommon):
    def __init__(self, learning_rate=0.001, name: str='ada_grad') -> None:
        SMLFComponent.__init__(self)
        OptimizerCommon.__init__(self, learning_rate)
        self._name = name

    def update_weights(self, layer: Dense):

        layer.adagrad_weights_acc += np.square(layer.derivatives_wrt_weights)
        layer.adagrad_biases_acc += np.square(layer.derivatives_wrt_biases)

        weights_norm = np.divide(self.learning_rate, np.sqrt(layer.adagrad_weights_acc + 1e-7))
        biases_norm = np.divide(self.learning_rate, np.sqrt(layer.adagrad_biases_acc + 1e-7))

        layer.weights -= np.multiply(layer.derivatives_wrt_weights, weights_norm)
        layer.biases -= np.multiply(layer.derivatives_wrt_biases, biases_norm)


    def update_params(self, layer: BatchDenseNormalization):
        layer.gamma = layer.gamma - self.learning_rate * layer.derivatives_wrt_gamma
        layer.beta = layer.beta - self.learning_rate * layer.derivatives_wrt_beta

    '''
        SMLF Base Component overrides
    '''

    def name(self):
        return self._name

    def component_name(self):
        return 'AdagradOptmizer'

    def save_state():
        pass

    def to_json(self):
        json = dict({
            'name': self.name(),
            'component_name': self.component_name(),
            'learning_rate': self.learning_rate,
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
