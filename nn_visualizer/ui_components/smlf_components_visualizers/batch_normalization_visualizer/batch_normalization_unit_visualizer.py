from PyQt5.QtCore import *
from PyQt5.QtGui import QMouseEvent, QPainter
from PyQt5.QtWidgets import *

from nn_visualizer.geometry.circle import Circle
from nn_visualizer.geometry.point import Point
from nn_visualizer.geometry.utils import point_lies_on_circle
from nn_visualizer.ui_components.canvas_drawable import CanvasDrawable
from nn_visualizer.ui_components.clickable import clickable
from nn_visualizer.ui_components.smlf_components_visualizers.layer_layout_data import LayerVisualizerLayoutData


class BatchNormalizationUnitVisualizer(CanvasDrawable):
    def __init__(self, x, y, layout_data: LayerVisualizerLayoutData):
        CanvasDrawable.__init__(self)
        self.width = 300
        self.height = 300
        self.x = x
        self.y = y
        self.is_selected = False
        self.layout_data = layout_data

    def draw(self, painter: QPainter):
        unit_point = QPoint(int(self.x), int(self.y))
        radius = self.layout_data.get_unit_radius()
        painter.drawEllipse(unit_point, int(radius), int(radius))

    def on_mouse_press(self, event: QMouseEvent):
        x, y = event.x(), event.y()
        radius = self.layout_data.get_unit_radius()
        if point_lies_on_circle(Circle(self.x, self.y, radius), Point(x, y)):
            self.is_selected = not self.is_selected

    def sizeHint(self) -> QSize:
        return QSize(self.width, self.height)

    def size(self) -> QSize:
        return QSize(self.width, self.height)
