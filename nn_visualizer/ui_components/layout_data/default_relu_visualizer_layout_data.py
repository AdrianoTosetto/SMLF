from __future__ import annotations
from typing import Callable

import numpy as np
from PyQt5.QtGui import QPainter

from api.sequential.pipeline.pipeline_state import PipelineState
from nn_visualizer.ui_components.smlf_components_visualizers.layer_layout_data import LayerVisualizerLayoutData


class DefaultReLUVisualizerLayoutData(LayerVisualizerLayoutData):
    SPACE_BETWEEN_NODES_CENTER = 50
    PADDING_VERTICAL = 20
    PADDING_HORIZONTAL = 30
    PADDING_LEFT = PADDING_RIGHT = PADDING_HORIZONTAL // 2
    PADDING_TOP = PADDING_BOTTOM = PADDING_VERTICAL // 2
    NODE_RADIUS = 15
    NODE_DIAM = NODE_RADIUS * 2
    SPACE_BETWEEN_NODES = SPACE_BETWEEN_NODES_CENTER - (2 * NODE_RADIUS)
    WIDTH = NODE_RADIUS + 2 * PADDING_HORIZONTAL

    def __init__(self, units, index: int, origin_x: float = 0,
                previous_layer_units_origins: list[tuple[float, float]] = [],
                next_layer_units_origins: list[tuple[float, float]] = []):
        LayerVisualizerLayoutData.__init__(self)
        self.index = index
        self._height = (units - 1) * self.SPACE_BETWEEN_NODES_CENTER + self.NODE_DIAM + 2*self.PADDING_VERTICAL
        self.pipeline_state = PipelineState.FORWARD


        self.origin = (origin_x, (500 - self.height()) / 2)
        self.previous_layer_units_origins = previous_layer_units_origins
        self.next_layer_units_origins = next_layer_units_origins
        
        self.input_units_origins = []
        self.output_units_origins = []

        self._padding = (
            self.PADDING_LEFT,
            self.PADDING_TOP,
            self.PADDING_RIGHT,
            self.PADDING_BOTTOM,
        )

        _, y = self.origin
        y += self.NODE_RADIUS + self.PADDING_VERTICAL
        x = self.origin[0] + (self.width() / 2)
        self._units_origins = [(x, y + i * self.SPACE_BETWEEN_NODES_CENTER) for i in range(units)]
        self.draw_forward_connections_strategy = None
        self.draw_backward_connections_strategy = None
        self.backward_derivatives = None

    def height(self) -> float:
        return self._height

    def width(self) -> float:
        return self.NODE_RADIUS + 2 * self.PADDING_HORIZONTAL

    def get_units_origins(self) -> list[tuple[float, float]]:
        return self._units_origins

    def set_origin(self, origin: tuple[float, float]) -> LayerVisualizerLayoutData:
        self.origin = origin

        return self

    def get_origin(self) -> tuple[float, float]:
        return self.origin

    def get_units_origins(self) -> list[tuple[float, float]]:
        return self._units_origins

    def set_units_origins(self, origins: list[tuple[float, float]]):
        self._units_origins = origins

    def get_input_units_origins(self) -> list(tuple[float, float]):
        return self.input_units_origins

    def set_input_units_origins(self, origins: list[tuple[float, float]]):
        self.input_units_origins = origins

    def get_output_units_origins(self) -> list[tuple[float, float]]:
        return self.output_units_origins

    def set_output_units_origins(self, origins: list[tuple[float, float]]):
        self.output_units_origins = origins

    def get_unit_radius(self) -> float:
        return self.NODE_RADIUS

    def get_padding(self) -> tuple[float, float, float, float]:
        return (
            self.PADDING_TOP,
            self.PADDING_RIGHT,
            self.PADDING_BOTTOM,
            self.PADDING_LEFT,
        )

    def get_backpropagation_derivatives(self) -> np.ndarray:
        return self.backward_derivatives

    def set_backpropagation_derivatives(self, derivatives: np.ndarray):
        self.backward_derivatives = derivatives

    def get_draw_forward_connections_strategy(self) -> Callable[[QPainter, LayerVisualizerLayoutData, int], None]:
        return self.draw_forward_connections_strategy

    def set_draw_forward_connections_strategy(self, strategy: Callable[[QPainter, LayerVisualizerLayoutData, int], None]):
        self.draw_forward_connections_strategy = strategy

    def get_draw_backward_connections_strategy(self) -> Callable[[QPainter, LayerVisualizerLayoutData, int], None]:
        return self.draw_backward_connections_strategy

    def set_draw_backward_connections_strategy(self, strategy: Callable[[QPainter, LayerVisualizerLayoutData, int], None]):
        self.draw_backward_connections_strategy = strategy

    def get_pipeline_state(self) -> PipelineState:
        return self.pipeline_state

    def set_pipeline_state(self, state: PipelineState):
        self.pipeline_state = state
