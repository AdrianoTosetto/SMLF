import json
from enum import Enum
from typing import Union

import numpy as np
from sklearn.utils import shuffle

from api.batch_dense_normalization import BatchDenseNormalization
from api.common_layer import CommonLayer
from api.internal.builders.json_builder.sfml_component_json_builder import SFMLComponentJsonBuilder
from api.internal.smlf_component import SMLFComponent
from api.layers.activation.softmax import (
    Softmax,
    SoftmaxCategoricalCrossEntropyOptimization,
)
from api.layers.dense import Dense
from api.layers.dense_l1_regularized import DenseL1Regularized
from api.layers.dense_l2_regularized import DenseL2Regularized
from api.layers.inv_dropout import InvDropout
from api.layers.softmax_categorical_cross_entropy_layer import SoftmaxCategoricalCrossEntropyLayer
from api.losses.binary_cross_entropy import BinaryCrossEntropy
from api.losses.categorical_cross_entropy import CategoricalCrossEntropy
from api.losses.common_loss import CommonLoss
from api.losses.regression.mean_squared_loss import MeanSquareError
from api.metrics.metric import Metric
from api.optimizers.adagrad_optimizer import AdagradOptmizer
from api.optimizers.optimizer_common import OptimizerCommon
from api.optimizers.stochastic_gradient_descent import StochasticGradientDescent


'''
    internal imports
'''


'''
    api imports
'''



def binary_cross_entropy_factory(units):
    return BinaryCrossEntropy(units)

def categorical_cross_entropy(units):
    return CategoricalCrossEntropy(units)

loss_factory = {
    'binary_cross_entropy': binary_cross_entropy_factory,
    'categorical_cross_entropy': categorical_cross_entropy,
}

def _build_sgd(learning_rate: float = 0.01):
    return StochasticGradientDescent(learning_rate=learning_rate)

def _build_adagrad(learning_rate: float = 0.01):
    return AdagradOptmizer(learning_rate=learning_rate)

loss_builder = dict({
    'sgd': _build_sgd,
    'ada_grad': _build_adagrad,
})

class SequentialModel(SMLFComponent):
    def __init__(self, layers: list[CommonLayer] = [], optimizer: str | OptimizerCommon = 'sgd',
                        loss: str | CommonLoss = 'binary_cross_entropy',
                        metrics: list[Metric] = [],
                        name: str = 'sequential_model',
                        learning_rate = 0.01,
        ):

        SMLFComponent.__init__(self)
        self.name = name

        self.layers = layers
        self.metrics = metrics

        self._build_loss(loss, self.layers[-1].units)
        self._build_optimizer(optimizer, learning_rate)

        self.has_softmax_cross_categorical_optimization = isinstance(self.layers[-1], Softmax) and isinstance(self.loss, CategoricalCrossEntropy)

        if (self.has_softmax_cross_categorical_optimization):
            self.layers[-1] = SoftmaxCategoricalCrossEntropyOptimization(self.layers[-1].units)

    '''
        builders/helpers methods
    '''
    def _build_loss(self, loss: str | CommonLoss, units: int):
        if isinstance(loss, str):
            self.loss = self._build_loss_from_name(loss, units)
        elif isinstance(loss, CommonLoss):
            self.loss = loss
        else:
            raise Exception('Invalid loss object')

    def _build_loss_from_name(self, name: str, units: int):
        if (name == 'binary_cross_entropy'):
            return BinaryCrossEntropy(units)
        if (name == 'mse' or name == 'mean_squared_error'):
            return MeanSquareError(units)
        if (name == 'categorical_cross_entropy'):
            return CategoricalCrossEntropy(units)

        raise Exception('Invalid loss name')

    def _build_optimizer_from_name(self, name:str, learning_rate):
        if (name == 'sgd'):
            return StochasticGradientDescent(learning_rate=learning_rate)
        if (name == 'ada_grad'):
            return AdagradOptmizer(learning_rate=learning_rate)

        raise Exception('Invalid optmizer name')

    def _build_optimizer(self, optimizer: str | OptimizerCommon, learning_rate):
        if isinstance(optimizer, OptimizerCommon):
                self.optimizer = optimizer
        elif isinstance(optimizer, str):
            self.optimizer = self._build_optimizer_from_name(optimizer, learning_rate)
        else:
            raise Exception('Invalid optmizer object')

    '''
        ==================================================
    '''

    '''
        return accuracy
    '''

    def evaluate(self, batch_input, targets):
        for metric in self.metrics:
            predictions = self.predict(batch_input)
            metric.update_state(predictions, targets)
            
    def predict(self, batch):
        return self._predict_forward(batch)

    def _set_loss(self, loss: str | CommonLoss = 'binary_cross_entropy', units: int = 0):
        if isinstance(loss, str):
            if loss_factory['loss'] is None:
                raise Exception('Invalid loss name')
            self.loss = loss_factory['loss'](units)
        elif isinstance(loss, CommonLoss):
            self.loss = loss
        else:
            raise Exception('Invalid loss object')

    def add_layer(self, layer):
        self.layers.append(layer)

    def _predict_forward(self, batch):
        if (self.has_softmax_cross_categorical_optimization):
            return self._softmax_cross_entropy_optimization_predict_forward(batch)
        return self._default_predict_forward(batch)

    def _default_predict_forward(self, batch: np.ndarray):
        input = batch
        for layer in self.layers:
            layer.forward(input, mode='predict')
            input = layer.output

        return input

    def _softmax_cross_entropy_optimization_predict_forward(self, batch: np.ndarray):
        input = batch

        for layer in self.layers[0:-1]:
            layer.forward(input, mode='predict')
            input = layer.output

        self.layers[-1].predict_forward(input)

        return self.layers[-1].output

    '''
        default implementations for forward and backward
    '''
    def _default_forward(self, batch: np.ndarray, targets: np.ndarray):
        reg_loss = .0
        input = batch

        for layer in self.layers:
            if isinstance(layer, SoftmaxCategoricalCrossEntropyLayer):
                layer.forward(input, targets)
            else:
                layer.forward(input)
            if isinstance(layer, DenseRegularizedDecorator):
                reg_loss += layer.regularization_term_weights() + layer.regularization_term_biases()
            input = layer.output

        self.loss.forward(input, targets)

    def _default_backward(self, targets: np.ndarray):
        self.loss.backward(targets)
        backward_derivatives = self.loss.derivatives_wrt_inputs

        for layer in self.layers[::-1]:
            layer.backward(backward_derivatives)
            backward_derivatives = layer.derivatives_wrt_inputs
    '''
        begin softmax + cross entropy optimization
        last layer = softmax and loss = cross entropy optimization
    '''
    def softmax_cross_categorical_optimization_forward(self, batch: np.ndarray, targets: np.ndarray):
        reg_loss = .0
        input = batch
        for layer in self.layers[0:-1]:
            layer.forward(input)
            if isinstance(layer, DenseL2Regularized) or isinstance(layer, DenseL1Regularized):
                reg_loss += layer.regularization_term_weights() + layer.regularization_term_biases()
            input = layer.output
    
        self.layers[-1].forward(input, targets)
        input = self.layers[-1].output

    def softmax_cross_categorical_optimization_backward(self, targets: np.ndarray):
        self.layers[-1].backward(targets)
        backward_derivatives =  self.layers[-1].derivatives_wrt_inputs.copy()

        for layer in self.layers[::-1][1:]:
            layer.backward(backward_derivatives)
            backward_derivatives = layer.derivatives_wrt_inputs.copy()

    '''
        end softmax + cross entropy optimization
    '''
    def _forward(self, batch, targets):
        if (self.has_softmax_cross_categorical_optimization):
            self.softmax_cross_categorical_optimization_forward(batch, targets)
        else:
            self._default_forward(batch, targets)

    def _backward(self, targets):
        if self.has_softmax_cross_categorical_optimization:
            self.softmax_cross_categorical_optimization_backward(targets)
        else:
            self._default_backward(targets)

    def _update_params(self):
        for layer in self.layers[::-1]:
            if isinstance(layer, Dense):
                self.optimizer.update_weights(layer)
            if isinstance(layer, BatchDenseNormalization):
                self.optimizer.update_params(layer)

    def fit(self, dataset, targets, epochs = 1, batch_size = 100):
        samples = len(dataset)

        for epoch in range(epochs):
            dataset, targets = shuffle(dataset, targets)
            loss = .0
            for i in range(0, samples, batch_size):
                low_index = i
                high_index = np.minimum(i+batch_size, samples)
                batch = dataset[low_index: high_index]
                batch_targets = targets[low_index: high_index]

                self._forward(batch, batch_targets)
                if self.has_softmax_cross_categorical_optimization:
                    loss += self.layers[-1].loss
                else:
                    loss += self.loss.loss

                self._backward(batch_targets)
                self._update_params()

            # if epoch % 100 == 0:
            str_msg = 'Epoch: {epoch:d}, Loss: {loss:.7f}'.format(epoch=epoch, loss=loss)
            print(str_msg)


    '''
        SMLF Base Compenent overrides
    '''

    def name(self,) -> str:
        return self.name

    def component_name(self) -> str:
        return 'SequentialModel'

    def to_json(self,):
        layers_json_list = list(map(lambda layer: layer.to_json(), self.layers))

        return dict({
            'name': self.name,
            'layers': layers_json_list,
            'loss': self.loss.to_json(),
        })

    @classmethod
    def from_json(cls, json_object: dict):
        name = json_object['name']
        layer_json_list = json_object['layers']
        loss_json = json_object['loss']

        layers = list(map(lambda json_layer: SFMLComponentJsonBuilder.build(json_layer), layer_json_list))

        return cls(layers, loss=SFMLComponentJsonBuilder.build(loss_json))

    @classmethod
    def from_string_json(cls, json_str: str):
        import json
        json_model = json.loads(json_str)

        return cls.from_json(json_model)

    def to_string_json(self):        
        return json.dumps(self.to_json())
