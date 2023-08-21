import numpy as np

from api.internal.smlf_component import SMLFComponent
from api.losses.common_loss import CommonLoss


class MeanSquareError(SMLFComponent, CommonLoss):
    def __init__(self, units, name: str  = 'mean_squared_error'):
        self.units = units
        self.name = name

        SMLFComponent.__init__(self)
        CommonLoss.__init__(self, units, units)

    def forward(self, batch_inputs: np.ndarray, targets: np.ndarray):
        self.samples = batch_inputs.shape[0]
        self.differences = batch_inputs - targets
        self.output = np.power(self.differences, 2)
        self.loss = np.sum(self.output) / self.samples

    def backward(self, targets: np.ndarray):
        self.derivatives_wrt_inputs = np.divide(2 * self.differences, self.samples)

    '''
        SMLF Base Compenent overrides
    '''

    def name(self,) -> str:
        return self.name

    def component_name(self) -> str:
        return 'MeanSquaredError'

    def to_json(self):
        return dict({
            'name': self.name,
            'component_name': self.component_name(),
            'units': self.ninputs,
        })

    @classmethod
    def from_json(cls, json_value: dict):
        name = json_value['name']
        units = json_value['units']

        return cls(units, name=name)
