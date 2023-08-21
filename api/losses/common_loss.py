from abc import ABC, abstractmethod

import numpy as np


class CommonLoss:
    def __init__(self, ninputs, noutputs) -> None:
        self.ninputs = ninputs
        self.noutputs = noutputs
        
    @abstractmethod
    def forward(self, batch_input: np.ndarray) -> None:
        pass

    @abstractmethod
    def backward(self, backward_derivatives: np.ndarray) -> None:
        pass

    @abstractmethod
    def get_loss() -> np.float64:
        pass
