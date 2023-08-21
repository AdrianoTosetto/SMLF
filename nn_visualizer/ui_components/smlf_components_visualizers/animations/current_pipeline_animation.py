import copy
import typing

import PyQt5.QtWidgets
from PyQt5.QtCore import QObject, QVariantAnimation
from PyQt5.QtWidgets import QWidget

from nn_visualizer.geometry.line_segment import LineSegment
from nn_visualizer.geometry.point import Point
from nn_visualizer.geometry.polygon import Polygon
from nn_visualizer.geometry.utils import (
    point_lies_on_circle,
    point_lies_on_polygon,
    trace_path,
    tracing_current_origin,
)


class CurrentPipelineAnimation(QVariantAnimation):
    def __init__(self, parent: QWidget = None, polygon: Polygon = None, update = None) -> None:
        self.polygon = polygon
        self._parent = parent
        self.update_callback = update
        QVariantAnimation.__init__(self,
            parent,
            valueChanged=self._animate,
            startValue=0.0,
            endValue=1.0,
            duration=2500,)
        self.finished.connect(self.restart)
        self._segments_state = ([], [])

    def _animate(self, value):
        (path_origin, index) = tracing_current_origin(self.polygon, value)
        segments1 = trace_path(self.polygon, path_origin, 100, index)

        if value > 0.5:
            value -= 0.5
        else:
            value += 0.5

        (path_origin, index) = tracing_current_origin(self.polygon, value)
        segments2 = trace_path(self.polygon, path_origin, 100, index)

        self._segments_state = (segments1, segments2)

        self.update_callback()

    def get_state(self):
        ret = self._segments_state[0]
        ret.extend(self._segments_state[1])

        return ret

    def restart(self):
        if not self.state():
            self.setStartValue(0.0)
            self.setEndValue(1.0)
            self.start()
