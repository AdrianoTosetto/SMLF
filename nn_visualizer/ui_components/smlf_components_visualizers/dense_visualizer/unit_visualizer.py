import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import QMouseEvent, QPainter, QPen
from PyQt5.QtWidgets import *

from nn_visualizer.geometry.circle import Circle
from nn_visualizer.geometry.point import Point
from nn_visualizer.geometry.utils import point_lies_on_circle
from nn_visualizer.ui_components.canvas_drawable import CanvasDrawable
from nn_visualizer.ui_components.clickable import clickable
from nn_visualizer.ui_components.smlf_components_visualizers.layer_layout_data import LayerVisualizerLayoutData


class VisualizerUnit(CanvasDrawable):
    def __init__(self, x = -1, y = -1, layout_data: LayerVisualizerLayoutData = None, index: int = 0):
        CanvasDrawable.__init__(self)
        self.x = x
        self.y = y
        self.selected = False
        self.layout_data = layout_data
        self.index = index
        # self.output_origin = self.layout_data.get_output_units_origins[index]

    def draw(self, painter: QPainter):
        painter.save()
        unit_point = QPoint(int(self.x), int(self.y))
        radius = self.layout_data.get_unit_radius()

        painter.drawEllipse(unit_point, radius, radius)

        if self.selected:
            input_units_origins = self.layout_data.get_input_units_origins()
            for input_origin in input_units_origins:
                input_origin_x, input_origin_y = input_origin

                a = unit_point
                b = QPoint(int(input_origin_x), int(input_origin_y))

                painter.drawLine(a, b)

            input_output_origins = self.layout_data.get_output_units_origins()
            import numpy as np

            derivatives = self.layout_data.get_backpropagation_derivatives()
            if derivatives is None:
                return
            
            max = np.max(derivatives)
            min = np.min(derivatives)

            for index, output_origin in enumerate(input_output_origins[self.index]):
                output_origin_x, output_origin_y = output_origin

                a = unit_point
                b = QPoint(int(output_origin_x), int(output_origin_y))

                from scipy.interpolate import interp1d
                # line_width = interp1d([min, max],[1,10])
                line_width = 1
                # print(line_width(derivatives[self.index][index]))

                painter.setPen(QPen(Qt.green, line_width))
                painter.drawLine(a, b)
        painter.restore()

    def on_mouse_press(self, event: QMouseEvent):
        x, y = event.x(), event.y()
        radius = self.layout_data.get_unit_radius()
        if point_lies_on_circle(Circle(self.x, self.y, radius), Point(x, y)):
            self.selected = not self.selected

    def sizeHint(self) -> QSize:
        return QSize(self.width, self.height)

    def size(self) -> QSize:
        return QSize(self.width, self.height)
