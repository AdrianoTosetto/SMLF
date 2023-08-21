from abc import abstractclassmethod
from enum import Enum

import numpy as np

from api.sequential.pipeline.pipeline_mode import PipelineMode
from api.sequential.pipeline.pipeline_state import PipelineState


class PipelineContext():

    @abstractclassmethod
    def get_state(self) -> PipelineState:
        pass

    @abstractclassmethod
    def set_state(self, state: PipelineState) -> None:
        pass

    @abstractclassmethod
    def set_current(self, index: int) -> None:
        pass

    @abstractclassmethod
    def get_current_input(self) -> np.ndarray:
        pass

    @abstractclassmethod
    def set_current_input(self, input: np.ndarray) -> None:
        pass

    @abstractclassmethod
    def set_current_backward_derivatives(self, derivatives: np.ndarray) -> None:
        pass

    @abstractclassmethod
    def get_current_backward_derivatives(self) -> np.ndarray:
        pass

    @abstractclassmethod
    def get_learning_rate(self) -> float:
        pass

    @abstractclassmethod
    def set_learning_rate(self, learning_rate: float) -> None:
        pass
