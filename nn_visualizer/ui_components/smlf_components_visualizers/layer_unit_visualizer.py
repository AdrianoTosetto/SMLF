import enum
from enum import Enum

import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import QMouseEvent, QPainter, QPen
from PyQt5.QtWidgets import *

from api.sequential.pipeline.pipeline_state import PipelineState
from nn_visualizer.geometry.circle import Circle
from nn_visualizer.geometry.point import Point
from nn_visualizer.geometry.utils import point_lies_on_circle
from nn_visualizer.ui_components.canvas_drawable import CanvasDrawable
from nn_visualizer.ui_components.clickable import clickable
from nn_visualizer.ui_components.smlf_components_visualizers.layer_layout_data import LayerVisualizerLayoutData
from nn_visualizer.ui_components.smlf_components_visualizers.strategies import draw_single_connected_backward_connection


class Showing(Enum):
    SHOWING_FORWARD_CONNECTIONS = 0,
    SHOWING_BACKWARD_CONNECTIONS = 1,
    SHOWING_BACKWARD_CONNECTIONS_IN_PATH = 2,

class LayerVisualizerUnit(CanvasDrawable):
    def __init__(self, layout_data: LayerVisualizerLayoutData = None, index: int = 0, on_node_selection_callback = None):
        CanvasDrawable.__init__(self)
        self.selected = False
        self.layout_data = layout_data
        self.index = index
        self.on_node_selection_callback = on_node_selection_callback
        self.showing_info = Showing.SHOWING_FORWARD_CONNECTIONS

    def draw(self, painter: QPainter):
        painter.save()
        (x, y) = self.layout_data.get_units_origins()[self.index]
        unit_radius = self.layout_data.get_unit_radius()

        unit_point = QPoint(int(x), int(y))
        radius = self.layout_data.get_unit_radius()

        painter.drawEllipse(unit_point, radius, radius)

        draw_strategy = self.layout_data.get_draw_backward_connections_strategy()
        if self.showing_info == Showing.SHOWING_BACKWARD_CONNECTIONS_IN_PATH and draw_strategy is not None:
                draw_strategy(painter, self.layout_data, self.index)

        if self.selected:
            # draw_strategy = self.layout_data.get_draw_forward_connections_strategy()


            # if self.layout_data.get_pipeline_state() == PipelineState.FORWARD and draw_strategy is not None:
            #     draw_strategy(painter, self.layout_data, self.index)


            draw_strategy = self.layout_data.get_draw_backward_connections_strategy()

            # if self.layout_data.get_pipeline_state() == PipelineState.BACKWARD and draw_strategy is not None:
            #     draw_strategy(painter, self.layout_data, self.index)
            if self.showing_info == Showing.SHOWING_BACKWARD_CONNECTIONS_IN_PATH and draw_strategy is not None:
                draw_strategy(painter, self.layout_data, self.index)
                          

        painter.restore()

    def on_mouse_press(self, event: QMouseEvent):
        x, y = event.x(), event.y()
        (unit_x, unit_y) = self.layout_data.get_units_origins()[self.index]
        radius = self.layout_data.get_unit_radius()

        if point_lies_on_circle(Circle(unit_x, unit_y, radius), Point(x, y)):
            self.selected = not self.selected
            if self.selected:
                self.on_node_selection_callback(self.index)

    def sizeHint(self) -> QSize:
        return QSize(self.width, self.height)

    def size(self) -> QSize:
        return QSize(self.width, self.height)
