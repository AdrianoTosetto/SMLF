from abc import ABC, abstractmethod

import numpy as np


'''
    mode = train | test | predict
'''

class CommonLayer:
    def __init__(self, ninputs, noutputs) -> None:
        self.ninputs = ninputs
        self.noutputs = noutputs
        
    @abstractmethod
    def forward(self, batch_input: np.ndarray, mode: str = 'train') -> None:
        pass

    @abstractmethod
    def backward(self, backward_derivatives: np.ndarray) -> None:
        pass
