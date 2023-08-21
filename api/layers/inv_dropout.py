import numpy as np

from api import common_layer


class InvDropout(common_layer.CommonLayer):
    def __init__(self, units: int, dropout_rate: float) -> None:
        super().__init__(units, units)
        self.dropout_rate = dropout_rate
        self.dropout_distribution = np.zeros((1, units))
        self.units = units

    def forward(self, batch_input: np.ndarray, mode: str = 'train') -> None:
        if (mode == 'train'):
            p = 1 - self.dropout_rate
            mask = np.random.rand(batch_input.shape[0], batch_input.shape[1]) < p
            # mask = np.random.binomial(1, p, size=(batch_input.shape[0], batch_input.shape[1]))
            self.output = mask * batch_input * (1.0 / p)
        elif (mode == 'predict'):
            self.output = batch_input
        else:
            raise Exception('Invalid mode')

    def backward(self, backward_derivatives: np.ndarray) -> None:
        self.derivatives_wrt_inputs = np.multiply(backward_derivatives, self.dropout_distribution)

    '''
        SMLF Base Compenent overrides
    '''
    def name(self):
        return self.name

    def component_name(self):
        return 'Dropout'

    def to_json(self,):
        dropout_distribution_values = list(self.dropout_distribution.reshape(1, -1))

        json = dict({
            'name': self.name,
            'class_name': self.component_name(),
            'units': self.ninputs,
            'dropout_rate': self.dropout_rate,
            'dropout_distribution': dropout_distribution_values
        })

        return json

    @classmethod
    def from_json(cls, json_object: dict):
        name = json_object['name']
        units = int(json_object['inits'])
        dropout_rate = float(json_object['dropout_rate'])
        dropout_distribution_values = json_object['dropout_distribution_values']

        ret = cls(units, dropout_rate)
        cls.dropout_distribution = np.array(dropout_distribution_values).reshape(1, -1)

        return ret

    def to_string_json(self):
        return str(self.to_json())

    def from_string_json():
        pass
