from api.common_layer import CommonLayer


class ActivationCommon(CommonLayer):
    def __init__(self, units: int):
        self.units = units
        super().__init__(ninputs=units, noutputs=units)
