import numpy as np

from api.metrics.metric import Metric


class CategoricalCrossEntropy(Metric):
    def __init__(self, name='binary_cross_entropy_metric'):
        self.name = name
        self.metric_value = .0

    def update_state(self, batch_predictions: np.ndarray, batch_labels: np.ndarray, sample_weight: np.ndarray = None):
        self.metric_value = np.sum(-np.log(np.clip(batch_predictions, 1e-07, 1 - 1e-07)) * batch_labels) / batch_labels.shape[0]

    def result(self):
        return self.metric_value

    def reset(self):
        self.metric_value = .0
