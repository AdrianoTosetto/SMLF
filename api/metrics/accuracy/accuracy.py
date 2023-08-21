import numpy as np


class Accuracy:
    def __init__(self, output_shape: np.shape, ):
        self.output_shape = output_shape

    def update_state(self, batch_predictions: np.ndarray, batch_labels: np.ndarray, sample_weight: np.ndarray = None):
            if (sample_weight is None):
                pass
                

    def _check_batch_size(self, shape: np.shape):
        pass
