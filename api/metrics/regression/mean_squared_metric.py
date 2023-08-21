import numpy as np


class MeanSquaredMetric:
    def __init__(self, units) -> None:
        self.regression_values: np.zeros((units))

    def forward(self, ouputs: np.ndarray, targets: np.array, sample_weights: np.array = np.array([1.])):
        errors = np.sum(np.power(ouputs - targets, 2), axis=1)

        if (len(sample_weights) == 1):
            self.output = errors
        else:
            self.output = np.multiply(errors, sample_weights)
