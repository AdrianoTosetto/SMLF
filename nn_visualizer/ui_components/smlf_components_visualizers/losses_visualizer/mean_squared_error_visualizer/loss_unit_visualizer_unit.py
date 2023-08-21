from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from nn_visualizer.ui_components.canvas_drawable import CanvasDrawable
from nn_visualizer.ui_components.smlf_components_visualizers.layer_unit_visualizer import LayerVisualizerUnit


class LossVisualizerUnit(LayerVisualizerUnit):
    UNIT_RADIUS = 20

    def __init__(self, x = -1, y = -1, input_unit_origins = []):
        LayerVisualizerUnit.__init__(self)
        self.x = x
        self.y = y
        self.width = 300
        self.height = 300
        self.is_selected = False
        self.input_units_origins = input_unit_origins

    def paint(self, painter: QPainter):
        painter.drawEllipse(self.x, self.y, 30, 30)

    def draw(self, painter: QPainter):
        painter.drawEllipse(QPoint(int(self.x), int(self.y)), LossVisualizerUnit.UNIT_RADIUS, LossVisualizerUnit.UNIT_RADIUS)

    def on_mouse_press(self, event: QMouseEvent):
        pass

    def get_padding(self, padding) -> tuple[int, int, int, int]:
        return super().get_padding(padding)

    def sizeHint(self) -> QSize:
        return QSize(self.width, self.height)

    def size(self) -> QSize:
        return QSize(self.width, self.height)
