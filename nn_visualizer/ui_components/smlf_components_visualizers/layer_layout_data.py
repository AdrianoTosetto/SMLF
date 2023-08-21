from __future__ import annotations
from abc import abstractmethod
from typing import Callable

import numpy as np
from PyQt5.QtGui import QPainter

from api.sequential.pipeline.pipeline_state import PipelineState


class LayerVisualizerLayoutData():
    def __init__(self):
        pass

    @abstractmethod
    def width(self) -> float:
        pass

    @abstractmethod
    def height(self) -> float:
        pass

    @abstractmethod
    def get_padding(self) -> tuple[float, float, float, float]:
        pass

    @abstractmethod
    def get_units_origins(self) -> list[tuple[float, float]]:
        pass

    @abstractmethod
    def set_units_origins(self, origins: list[tuple[float, float]]):
        pass

    @abstractmethod
    def get_input_units_origins(self) -> list(tuple[float, float]):
        pass

    @abstractmethod
    def set_input_units_origins(self, origins: list[tuple[float, float]]):
        pass

    @abstractmethod
    def get_output_units_origins(self) -> list[tuple[float, float]]:
        pass

    @abstractmethod
    def set_output_units_origins(self, origins: list[tuple[float, float]]):
        pass

    @abstractmethod
    def get_origin(self) -> tuple[float, float]:
        pass

    @abstractmethod
    def set_origin(self, origin: tuple[float, float]):
        pass

    @abstractmethod
    def get_unit_radius(self) -> float:
        pass

    @abstractmethod
    def get_padding(self) -> tuple[float, float, float, float]:
        pass

    @abstractmethod
    def get_layer_is_selectable(self) -> bool:
        pass

    @abstractmethod
    def get_layer_unit_is_selectable(self) -> bool:
        pass

    @abstractmethod
    def set_layer_is_selectable(self, value: bool):
        pass

    @abstractmethod
    def set_layer_unit_is_selectable(self, value: bool):
        pass

    @abstractmethod
    def get_backpropagation_derivatives(self) -> np.ndarray:
        pass

    @abstractmethod
    def set_backpropagation_derivatives(self, derivatives: np.ndarray):
        pass

    @abstractmethod
    def set_draw_backward_connections_strategy(
            self,
            strategy: Callable[[QPainter, LayerVisualizerLayoutData, int], None]
        ):
        pass

    @abstractmethod
    def get_draw_backward_connections_strategy(self) -> Callable[[QPainter, LayerVisualizerLayoutData, int], None]:
        pass


    @abstractmethod
    def set_draw_forward_connections_strategy(
            self,
            strategy: Callable[[QPainter, LayerVisualizerLayoutData, int], None]
        ):
        pass

    @abstractmethod
    def get_draw_forward_connections_strategy(self) -> Callable[[QPainter, LayerVisualizerLayoutData, int], None]:
        pass

    @abstractmethod
    def get_pipeline_state(self) -> PipelineState:
        pass

    @abstractmethod
    def set_pipeline_state(self, state: PipelineState):
        pass
