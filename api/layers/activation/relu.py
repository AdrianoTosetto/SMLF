import numpy as np

from api import common_layer
from api.internal.smlf_component import SMLFComponent
from api.layers.activation.activation_common import ActivationCommon


class ReLU(ActivationCommon, SMLFComponent):
    def __init__(self, units: int = 1, name: str = 'relu'):
        ActivationCommon.__init__(self, units)
        SMLFComponent.__init__(self)
        self.name = name

    def forward(self, batch_input: np.ndarray, mode: str = 'train') -> None:
       self.output = np.maximum(batch_input, 0)
       self.zero_indexes = np.argwhere(self.output == 0)
       self.zero_indexes_col = self.zero_indexes[:, 0]
       self.zero_indexes_row = self.zero_indexes[:, 1]
    
    def backward(self, backward_derivatives: np.ndarray) -> None:
        self.derivatives_wrt_inputs = np.copy(backward_derivatives)
        self.derivatives_wrt_inputs[self.zero_indexes_col, self.zero_indexes_row] = 0

    '''
        SMLF Base Compenent overrides
    '''
    def name(self):
        return self.name
    
    def component_name(self):
        return 'ReLU'

    def to_json(self,):
        json = dict({
            'name': self.name,
            'component_name': self.component_name(),
            'units': self.units,
        })

        return json

    @classmethod
    def from_json(cls, json_object: dict):
        name = json_object['name']
        units = json_object['units']

        return cls(units, name=name)

    def from_string_json(json_layer: dict):
        pass

    @classmethod
    def to_string_json(self):        
        return str(self.to_json())
