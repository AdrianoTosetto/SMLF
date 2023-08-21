
import json

import PyQt5
from PyQt5 import QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtGui import QBrush, QPainter, QPen
from PyQt5.QtWidgets import *

from api.layers.dense import Dense
from nn_visualizer.application_context.application_context import ApplicationContext
from nn_visualizer.geometry.line_segment import LineSegment
from nn_visualizer.geometry.point import Point
from nn_visualizer.geometry.polygon import Polygon
from nn_visualizer.geometry.utils import (
    is_zero,
    line_segments_intersects,
    trace_path,
    tracing_current_origin,
)
from nn_visualizer.ui_components.canvas_drawable import CanvasDrawable
from nn_visualizer.ui_components.clickable import clickable


# from nn_visualizer.ui_components.smlf_components_visualizers.dense_visualizer.nn_visualizer_dense_layer import DenseVisualizer


class CanvasWidget(QWidget):
    def __init__(self, application_context: ApplicationContext, height = 0, width = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self.setSizePolicy(
            QSizePolicy.MinimumExpanding,
            QSizePolicy.Expanding
        )

        self.drawables: list[CanvasDrawable] = []
        self.mousePressEvent = self.mousePressEvent
        self.application_context = application_context
        self.legend = None

    def set_legend(self, legend: CanvasDrawable):
        self.legend = legend

    def _next_callback(self):
        self._animation.stop()

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.save()

        brush = QBrush()
        brush.setColor(QColor('white'))
        brush.setStyle(Qt.SolidPattern)
        rect = QRect(0, 0, painter.device().width(), painter.device().height())
        painter.fillRect(rect, brush)

        for drawable in self.drawables:
            drawable.draw(painter)

        if self.legend is not None:
            self.legend.draw(painter)

        painter.restore()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        for drawable in self.drawables:
            drawable.on_mouse_press(event)

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, event: QDropEvent) -> None:
        layer_name = json.loads(event.mimeData().text())['layer_name']
        if layer_name == 'Dense':
            self.application_context.add_dense_layer()

        if layer_name == 'Dropout':
            self.application_context.add_dropout_layer()

        if layer_name == 'BatchNormalization':
            self.application_context.add_batch_normalization_layer()

        if layer_name == 'ReLU':
            self.application_context.add_relu_layer()

        if layer_name == 'Sigmoid':
            self.application_context.add_sigmoid_layer()

        if layer_name == 'Softmax':
            self.application_context.add_softmax_layer()

        event.accept()
        self.update()

    def set_drawables(self, components):
        def set_parent(component):
            component.parent = self
            return component

        self.drawables = list(map(set_parent, components))
        # self.drawables = components

