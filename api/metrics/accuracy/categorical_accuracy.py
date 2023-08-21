import numpy as np


class CategoricalAccuracy():
    def __init__(self, name='binary_accuracy', threshold=0.5):
        self.name = name
        self.threshold = threshold

    def update_state(self, batch_predictions: np.ndarray, batch_labels: np.ndarray, sample_weight: np.ndarray = None):
            if (sample_weight is None):
                if not self._check_sizes(batch_predictions.shape, batch_labels.shape):
                    raise Exception('Invalid target shape: either use class or onehot enconding')
                encondig = self._get_targets_enconding(batch_labels.shape)
                if (encondig == 'onehot'):
                    tmp = np.argmax(batch_predictions, axis=1) == np.argmax(batch_labels, axis=1)
                    self.accuracy = np.sum(tmp,) / tmp.shape[0]

    def result(self):
        return self.accuracy                

    def _check_sizes(self, batch_shape: np.shape, targets_shape: np.shape) -> bool:
        if (targets_shape[1] != batch_shape[1] and targets_shape[1] != 1):
            return False
        return True

    def _get_targets_enconding(self, shape: np.shape):
        if shape[1] == 1:
            return 'class'
        return 'onehot'
