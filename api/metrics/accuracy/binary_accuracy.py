import numpy as np


class BinaryAccuracy():
    def __init__(self, name='binary_accuracy', threshold=0.5):
        self.name = name
        self.threshold = threshold

    def update_state(self, batch_predictions: np.ndarray, batch_labels: np.ndarray, sample_weight: np.ndarray = None):
        temp = np.absolute(batch_predictions - batch_labels)
        temp = np.where(temp <= self.threshold, 1, 0)

        samples = len(temp)
        if sample_weight is not None:
            samples = len(sample_weight)

        self.accuracy = np.sum(temp) / samples

    def result(self):
        return self.accuracy                

    def _check_batch_size(self, shape: np.shape):
        pass
