import numpy as np

from api.common_layer import CommonLayer
from api.internal.smlf_component import SMLFComponent


class Softmax(SMLFComponent, CommonLayer):
    def __init__(self, units: int, name: str = 'softmax'):
        SMLFComponent.__init__(self)
        CommonLayer.__init__(self, ninputs=units, noutputs=units)
        self.name = name
        self.units = units

    def forward(self, batch_input: np.ndarray) -> None:
        batch_input_normalized = batch_input - np.max(batch_input, axis=1, keepdims=True)
        exponentials = np.exp(batch_input_normalized)

        self.output = exponentials / np.sum(exponentials, axis=1, keepdims=True)

    def backward(self, backward_derivatives: np.ndarray) -> None:
        self.backprop_values = np.empty(shape=backward_derivatives.shape)

        for i in range(0, self.backprop_values.shape[0]):
            output = self.out[i]
            next_layer_backprop_value = backward_derivatives[i]
            matrix = self._generateBackpropMatrix(output)
            self.backprop_values[i] = np.matmul(matrix, next_layer_backprop_value)

    '''
        output: shape(1, ninputs), softmax output
        return: shape(ninputs, ninputs)

        example: [[a b c]] -> 
            [
                [a*(1-a), -a*b, -a*c],
                [-b*a, b*(1-b), -b*c],
                [-b*a, -a*b, c*(1-c)],
            ]
    '''

    def _generateBackpropMatrix(self, output: np.ndarray) -> np.ndarray:
        routput = output.reshape(-1, self.ninputs)
        matrix = -1 * np.matmul(routput.T, routput)
        matrix_diagonal = np.add(matrix.diagonal(), output)

        np.fill_diagonal(matrix, matrix_diagonal)

        return matrix

    '''
        SMLF Base Compenent overrides
    '''
    def name(self):
        return self.name

    def component_name(self):
        return 'Softmax'

    @classmethod
    def from_json(cls, json_value: dict):
        name = json_value['name']
        units = json_value['units']

        return cls(units, name=name)

    def to_json(self):
        return dict({
            'name': self.name,
            'component_name': self.component_name()
        })

class SoftmaxCategoricalCrossEntropyOptimization(Softmax):
    def __init__(self, units: int, name: str = 'softmax'):
        Softmax.__init__(self, units, name=name)

    def train_forward(self, batch_input: np.ndarray, batch_targets: np.ndarray):
        Softmax.forward(self, batch_input)
        batch_size = batch_input.shape[0]
        error = -np.log(np.clip(self.output, 1e-7, 1 - 1e-7)) * batch_targets
        self.loss = np.sum(error) / batch_size
        # print('np clip = ', np.clip(self.output, 1e-7, 1 - 1e-7))

    def test_forward(self, batch_input: np.ndarray):
        Softmax.forward(self, batch_input)

    def predict_forward(self, batch_input: np.ndarray):
        Softmax.forward(self, batch_input)

    def forward(self, batch_input: np.ndarray, batch_targets: np.ndarray, mode: str = 'train') -> None:
        if (mode == 'train'):
            self.train_forward(batch_input, batch_targets)
        elif mode == 'test':
            self.test_forward(batch_input)
        elif mode == 'predict':
            self.predict_forward(batch_input)

    def backward(self, targets: np.ndarray) -> None:
        self.derivatives_wrt_inputs = (self.output - targets) / targets.shape[0]

    '''
        SMLF Base Compenent overrides
    '''
    def name(self):
        return self.name

    def component_name(self):
        return 'SoftmaxCategoricalCrossEntropyOptimization'

    @classmethod
    def from_json(cls, json_value: dict):
        name = json_value['name']
        units = json_value['units']

        return cls(units, name=name)

    def to_json(self):
        return dict({
            'name': self.name,
            'component_name': self.component_name()
        })
