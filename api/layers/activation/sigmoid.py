import numpy as np

from api.internal.smlf_component import SMLFComponent
from api.layers.activation.activation_common import ActivationCommon


class Sigmoid(ActivationCommon, SMLFComponent):
    def __init__(self, units: int, name: str = 'sigmoid') -> None:
        ActivationCommon.__init__(self, units)
        SMLFComponent.__init__(self)
        self.name = name

    def forward(self, batch_input: np.ndarray, mode: str = 'predict') -> None:
        # self.output = np.divide(1, np.clip(1 + np.exp(-batch_input), 1e-7, 1 - 1e-7))
        self.output = np.divide(1, 1 + np.exp(-batch_input))

    def backward(self, backward_derivatives: np.ndarray, mode: str = 'train') -> None:
        derivatives = np.multiply(self.output, (1 - self.output))
        self.derivatives_wrt_inputs = np.multiply(derivatives, backward_derivatives)

    '''
        SMLF Base Compenent overrides
    '''

    def name(self):
        return self.name

    def component_name(self):
        return 'Sigmoid'

    def to_json(self,):
        json = dict({
            'name': self.name,
            'class_name': self.component_name(),
            'units': self.units,
        })

        return json

    @classmethod
    def from_json(cls, json_value: dict):
        name = json_value['name']
        units = json_value['units']

        return cls(units, name=name)

    @classmethod
    def to_string_json(self):        
        return str(self.to_json())

    def from_string_json():
        pass
