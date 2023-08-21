import numpy as np

from api.internal.smlf_component import SMLFComponent
from api.losses.common_loss import CommonLoss


'''
    used for multi-label classification tasks
'''

class BinaryCrossEntropy(SMLFComponent, CommonLoss):
    def __init__(self, ninputs, name: str = 'binary_cross_entropy') -> None:
        self.ninputs = ninputs
        self.units = ninputs
        self.name = name

        SMLFComponent.__init__(self)

    def forward(self, batch_inputs: np.ndarray, targets: np.ndarray, mode = 'train'):
        samples_size = batch_inputs.shape[0]
        inputs_size = batch_inputs.shape[1]
        self.batch_inputs = np.clip(batch_inputs, 1e-7, 1 - 1e-7)

        log_batch_inputs = np.log(self.batch_inputs)
        log_one_minus_batch_inputs = np.log(1 - self.batch_inputs)

        self.output = -np.multiply(targets, log_batch_inputs) - np.multiply(1 - targets, log_one_minus_batch_inputs)

        self.loss = np.sum(self.output) / (samples_size * inputs_size)

    def backward(self, targets: np.ndarray):
        self.derivatives_wrt_inputs = -1*(np.divide(targets, self.batch_inputs) - np.divide(1 - targets, 1 - self.batch_inputs)) / len(targets)

    '''
        SMLF Base Compenent overrides
    '''

    def name(self,) -> str:
        return self.name

    def component_name(self) -> str:
        return 'BinaryCrossEntropy'

    def to_json(self):
        return dict({
            'name': self.name,
            'component_name': self.component_name(),
            'units': self.ninputs,
        })

    def from_json(cls, json_value: dict):
        name = json_value['name']
        units = json_value['units']

        return cls(units, name=name)
