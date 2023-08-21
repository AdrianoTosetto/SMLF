import numpy as np

from api.layers import dense


class OptimizerCommon:
    def __init__(self, learning_rate=1e-3) -> None:
        self.learning_rate = learning_rate
    

    def update_weights(self, layer: dense.Dense, derivatives: np.ndarray):
        pass
