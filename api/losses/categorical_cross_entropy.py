import numpy as np

from api.losses.common_loss import CommonLoss


class CategoricalCrossEntropy(CommonLoss):
    def __init__(self, units):
        self.units = units

    def forward(self, batch_inputs: np.ndarray, targets: np.ndarray):
        self.samples = batch_inputs.shape[0]
        self.targets = targets
        self.output = -np.log(batch_inputs) * targets
        self.loss = np.sum(self.output) / self.samples

    def backward(self, ):
        self.derivatives_wrt_inputs = self.output

    '''
        SMLF Base Compenent overrides
    '''

    def name(self,) -> str:
        return self.name

    def component_name(self) -> str:
        return 'CategoricalCrossEntropy'

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
