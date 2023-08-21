import numpy as np

from api.metrics.metric import Metric


class BinaryCrossEntropy(Metric):
    def __init__(self, name='binary_cross_entropy_metric'):
        self.name = name
        self.loss = .0

    def update_state(self, batch_predictions: np.ndarray, batch_labels: np.ndarray, sample_weight: np.ndarray = None):
        if (sample_weight is None):
            samples_size = batch_predictions.shape[0]
            inputs_size = batch_predictions.shape[1]
            self.batch_inputs = np.clip(batch_predictions, 1e-7, 1 - 1e-7)

            log_batch_inputs = np.log(self.batch_inputs)
            log_one_minus_batch_inputs = np.log(1 - self.batch_inputs)

            self.output = -np.multiply(batch_labels, log_batch_inputs) - np.multiply(1 - batch_labels, log_one_minus_batch_inputs)

            self.loss += np.sum(self.output) / (samples_size * inputs_size)
        else:
            # TODO: implement sampling weight
            pass

    def result(self):
        return self.loss

    def reset(self):
        self.loss = .0
