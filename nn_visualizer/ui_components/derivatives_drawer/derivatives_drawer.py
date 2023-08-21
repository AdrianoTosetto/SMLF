import PyQt5
from PyQt5.QtCore import QRect, QRectF, Qt
from PyQt5.QtGui import *
from PyQt5.QtGui import QBrush, QColor, QPainter
from PyQt5.QtWidgets import *

from nn_visualizer.ui_components.canvas_drawable import CanvasDrawable


class DerivativesDrawer(CanvasDrawable):
    def __init__(self,
            layer_index: int,
            node_index: int,
            next_layer_node_indexes: list[int],
            derivatives_colors: list[QColor],
        ):
        CanvasDrawable.__init__(self)
        self.layer_index = layer_index
        self.node_index = node_index
        self.next_layer_node_indexes = next_layer_node_indexes
        self.derivatives_colors = derivatives_colors
        self.x = 1000
        self.y = 40
        self.derivative_str_line_height = 25
        self.derivatives_strings = []
        self._set_strings()

    def _set_strings(self):
        next_layer_index = self.layer_index + 1
        strings: list[str] = []

        for node_index in self.next_layer_node_indexes:
            partial_derivative_str = '∂Layer{next_layer_index}Out{next_layer_node_index}/∂Layer{layer_index}Out{node_index}'\
                .format(
                    next_layer_index=next_layer_index,
                    next_layer_node_index=node_index,
                    layer_index=self.layer_index,
                    node_index=self.node_index,
                )

            strings.append(partial_derivative_str)

        self.derivatives_strings = strings

    def draw(self, painter: QPainter):
        painter.save()

        brush = QBrush()
        brush.setColor(QColor('#145DA0'))
        brush.setStyle(Qt.SolidPattern)
        rect = QRect(self.x, self.y, self.width(), self.height())
        painter.fillRect(rect, brush)

        painter.setPen(Qt.white)
        y_translation = 0
        
        for derivative_str in self.derivatives_strings:
            y = self.y + self.get_vertical_padding() + y_translation
            painter.drawText(
                QRectF(self.x, y, self.width() - 50, self.derivative_str_line_height),
                Qt.AlignCenter,
                derivative_str,
            )

            y_translation += self.derivative_str_line_height
    
        y_translation = 2.5

        for color in self.derivatives_colors:
            brush = QBrush()
            brush.setColor(color)
            brush.setStyle(Qt.SolidPattern)
            y = self.y + self.get_vertical_padding() + y_translation
            rect = QRect(self.x + self.width() - 50, int(y), 40, 20)
            painter.fillRect(rect, brush)
    
            y_translation += self.derivative_str_line_height

        painter.restore()

    def _draw_color_rect(self, painter: QPainter, x: int, y: int, color: QColor):
        painter.save()

        brush = QBrush()
        brush.setColor(color)
        brush.setStyle(Qt.SolidPattern)
        rect = QRect(x, y, 100, 100)
        painter.fillRect(rect, brush)

        painter.restore()

    def width(self) -> float:
        return 280

    def height(self) -> float:
        return len(self.derivatives_strings) * self.derivative_str_line_height + 2*self.get_vertical_padding()

    def get_horizontal_padding(self) -> tuple[int, int]:
        return 20

    def get_vertical_padding(self) -> tuple[int, int]:
        return 30
