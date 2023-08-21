from PyQt5.QtCore import *
from PyQt5.QtGui import QMouseEvent, QPainter
from PyQt5.QtWidgets import *

from nn_visualizer.geometry.circle import Circle
from nn_visualizer.geometry.point import Point
from nn_visualizer.geometry.utils import point_lies_on_circle
from nn_visualizer.ui_components.canvas_drawable import CanvasDrawable
from nn_visualizer.ui_components.clickable import clickable


class SigmoidVisualizerUnit(CanvasDrawable):
    def __init__(self, x = -1, y = -1, radius=1.0):
        CanvasDrawable.__init__(self)
        self.x = x
        self.y = y
        self.is_selected = False
        self.radius = radius
        self.selected = False

    def draw(self, painter: QPainter):
        unit_point = QPoint(int(self.x), int(self.y))
        painter.drawEllipse(QPoint(int(self.x), int(self.y)), self.radius, self.radius)

    def on_mouse_press(self, event: QMouseEvent):
        x, y = event.x(), event.y()
        if point_lies_on_circle(Circle(self.x, self.y, self.radius), Point(x, y)):
            self.is_selected = not self.is_selected

    def sizeHint(self) -> QSize:
        return QSize(self.width, self.height)

    def size(self) -> QSize:
        return QSize(self.width, self.height)
