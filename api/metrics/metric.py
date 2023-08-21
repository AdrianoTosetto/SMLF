from abc import abstractmethod

import numpy as np


class Metric:
    def __init__(self, name='binary_cross_entropy_metric'):
        self.name = name
        self.loss = .0

    @abstractmethod
    def update_state(self, batch_labels: np.ndarray, batch_predictions, sample_weight: np.ndarray = None):
        pass

    @abstractmethod
    def result(self):
        pass

    @abstractmethod
    def reset(self):
        pass
