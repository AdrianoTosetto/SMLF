import json

import numpy as np

from api import common_layer
from api.internal.builders.json_builder.sfml_component_json_builder import SFMLComponentJsonBuilder
from api.internal.smlf_component import SMLFComponent
from api.layers.activation.relu import ReLU
from api.layers.activation.sigmoid import Sigmoid


def _build_relu(units):
    return ReLU(units=units)

def _build_sigmoid(units):
    return Sigmoid(units=units)

activation_builder = dict({
    'relu': _build_relu,
    'sigmoid': _build_sigmoid,
})

class Dense(SMLFComponent, common_layer.CommonLayer):
    def __init__(self, ninputs, noutputs, name: str = 'Dense', init_algorithm='uniform',
                 activation: str | common_layer.CommonLayer=None,):
        
        SMLFComponent.__init__(self)
        common_layer.CommonLayer.__init__(self, ninputs, noutputs)
        self.name = name
        self.units = noutputs

        if (init_algorithm == 'uniform'):
            self.weights = np.random.uniform(-1, 1, size=(self.noutputs, self.ninputs))
        if (init_algorithm == 'random'):
            self.weights = np.random.randn(self.noutputs, self.ninputs)
        if (init_algorithm == 'zeroes'):
            self.weights = np.zeros(shape=(self.noutputs, self.ninputs))

        self.biases = np.zeros((1, self.noutputs))

        self.momentum_weights = np.zeros(shape=self.weights.shape)
        self.momentum_biases = np.zeros(shape=self.biases.shape)
        self.adagrad_weights_acc = np.zeros(shape=self.weights.shape)
        self.adagrad_biases_acc = np.zeros(shape=self.biases.shape)

        if isinstance(activation, common_layer.CommonLayer):
            self.activation = activation
        elif isinstance(activation, str):
            self.activation = self._build_activation(activation, self.noutputs)
        else:
            self.activation = None

    def forward(self, batch_input: np.ndarray, mode: str = 'train') -> None:
        self.batch_input = batch_input
        self.output = np.add(np.dot(self.batch_input, self.weights.T), self.biases)

        if isinstance(self.activation, common_layer.CommonLayer):
            self.activation.forward(self.output)
            self.output = self.activation.output
    
    def dense_backward(self, next_layer_backward: np.ndarray):
        self.derivatives_wrt_inputs = np.dot(next_layer_backward, self.weights)
        self.derivatives_wrt_weights = np.dot(next_layer_backward.T, self.batch_input)
        self.derivatives_wrt_biases = np.sum(next_layer_backward, axis=0, keepdims=True)
    
    def dense_activation_backward(self, next_layer_backward: np.ndarray):    
        self.activation.backward(next_layer_backward)
        self.derivatives_wrt_inputs = np.dot(self.activation.derivatives_wrt_inputs, self.weights)

        self.derivatives_wrt_weights = np.dot(self.activation.derivatives_wrt_inputs.T, self.batch_input)
        self.derivatives_wrt_biases = np.sum(self.activation.derivatives_wrt_inputs, axis=0, keepdims=True)

    def backward(self, next_layer_backward: np.ndarray) -> None:
        if (self.activation is None):
            self.dense_backward(next_layer_backward)
        else:
            self.dense_activation_backward(next_layer_backward)

    def _build_activation(self, activation_name, units: int):
        if (activation_builder[activation_name] is None):
            raise Exception('Invalid activation layer')
        return activation_builder[activation_name](units)

    '''
        SMLF Base Component overrides
    '''
    def name(self):
        return self.name

    def component_name(self):
        return 'Dense'

    def to_json(self,):
        serialized_weights = self.weights.reshape(1, -1).tolist()
        biases = self.biases.reshape(1, -1).tolist()

        json = dict({
            'name': self.name,
            'component_name': self.component_name(),
            'serialized_weights': serialized_weights,
            'serialized_biases': biases,
            'units': self.noutputs,
            'ninputs': self.ninputs,
            'noutputs': self.noutputs,
            'activation': None if not self.activation else self.activation.to_json()
        })

        return json

    @classmethod
    def from_json(cls, json_object: dict):
        activation_json = json_object['activation']
        name = json_object['name']
        serialized_weights = json_object['serialized_weights']
        serialized_biases = json_object['serialized_biases']
        units = json_object['units']
        ninputs = json_object['ninputs']

        weights = np.array(serialized_weights).reshape(units, ninputs)
        biases = np.array(serialized_biases).reshape(1, units)
        activation = None

        if activation_json is not None:
            activation = SFMLComponentJsonBuilder.build(activation_json)

        ret = cls(ninputs, units, init_algorithm='zeroes', activation=activation)

        ret.weights = weights
        ret.biases = biases

        return ret

    def to_string_json(self):
        return str(self.to_json())

    def from_string_json():
        pass

    def reset_state(self):
        self.momentum_weights = np.zeros(shape=self.weights.shape)
        self.momentum_biases = np.zeros(shape=self.biases.shape)
        self.adagrad_weights_acc = np.zeros(shape=self.weights.shape)
        self.adagrad_biases_acc = np.zeros(shape=self.biases.shape)
