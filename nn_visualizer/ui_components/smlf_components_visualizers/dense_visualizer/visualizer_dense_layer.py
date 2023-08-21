from __future__ import annotations

from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtGui import QMouseEvent, QPainter
from PyQt5.QtWidgets import *

from api.layers.dense import Dense
from nn_visualizer.geometry.line_segment import LineSegment
from nn_visualizer.geometry.point import Point
from nn_visualizer.geometry.polygon import Polygon
from nn_visualizer.geometry.utils import point_lies_on_polygon
from nn_visualizer.ui_components.canvas_drawable import CanvasDrawable
from nn_visualizer.ui_components.smlf_components_visualizers.animations.current_pipeline_animation import CurrentPipelineAnimation
from nn_visualizer.ui_components.smlf_components_visualizers.layer_layout_data import LayerVisualizerLayoutData
from nn_visualizer.ui_components.smlf_components_visualizers.layer_unit_visualizer import LayerVisualizerUnit
from nn_visualizer.ui_components.smlf_components_visualizers.strategies import draw_fully_connected_backward_connections


class DenseVisualizer(CanvasDrawable):
    def __init__(
            self,
            layout_data: LayerVisualizerLayoutData,
            parent = None,
            on_selection_callback = None,
            on_node_selection_callback = None,
            name: str = 'Dense',
            index: int = 0,
        ):

        CanvasDrawable.__init__(self)
        self.layout_data = layout_data
        self.name = name
        self.previous_layer_units_origins = []
        self.x, self.y = layout_data.get_origin()
        self._set_polygon()
        self.on_node_selection_callback = on_node_selection_callback
        self.drawable_units = self._build_drawable_units()
        self.selected = False
        self.is_current = False
        self.parent = parent
        self.on_selection_callback = on_selection_callback
        self.layout_data.set_draw_backward_connections_strategy(draw_fully_connected_backward_connections)

        self.animation = CurrentPipelineAnimation(parent=parent, polygon=self.polygon, update=self.update)
        self.animation.start()

    def _build_drawable_units(self):
        radius = self.layout_data.get_unit_radius()
        units_origins = self.layout_data.get_units_origins()

        return [LayerVisualizerUnit(self.layout_data, index, self.on_node_selection_callback) for (index, (x, y)) in enumerate(units_origins)]

    def units_origins(self):
        return self.layout_data.get_units_origins()

    def update(self):
        self.parent.update()

    def restart(self):
        if not self.animation.state():
            self.animation.setStartValue(0.0)
            self.animation.setEndValue(1.0)
            self.animation.start()

    def get_shape(self) -> Polygon:
        return self.polygon

    def _set_polygon(self) -> list[LineSegment]:
        self.polygon = Polygon.from_vertices(
            vertices=[
                Point(self.x, self.y),
                Point(self.x + self.layout_data.width(), self.y),
                Point(self.x + self.layout_data.width(), self.y + self.layout_data.height()),
                Point(self.x, self.y + self.layout_data.height()),
            ])

    def height(self) -> float:
        return self.layout_data.height()

    def width(self) -> float:
        return self.layout_data.width()

    def get_padding_left(self) -> int:
        padding_left, _, _, _ = self.layout_data.get_padding()

        return padding_left

    def get_padding_right(self) -> int:
        _, _, padding_right, _ = self.layout_data.get_padding()

        return padding_right

    def get_padding_top(self) -> int:
        _, padding_top, _, _ = self.layout_data.get_padding()

        return padding_top

    def get_padding_bottom(self) -> int:
        _, _, _, padding_bottom = self.layout_data.get_padding()

        return padding_bottom

    def draw(self, painter: QPainter):
        (x, y) = self.layout_data.get_origin()
        painter.drawText(QRectF(x, y, self.width(), -20), Qt.AlignCenter, self.name)

        for unit in self.drawable_units:
            unit.draw(painter)

        if self.is_current:
            segments = self.animation.get_state()
            for segment in segments:
                painter.drawLine(segment.to_QLine())

        painter.save()
        painter.setPen(Qt.red)

        if self.selected:
            for line_segment in self.get_shape().edges:
                line = line_segment.to_QLine()
                painter.drawLine(line)

        painter.restore()

    def on_mouse_press(self, event: QMouseEvent):
        p = Point(event.x(), event.y())
        if point_lies_on_polygon(self.polygon, p):
            for unit in self.drawable_units:
                unit.on_mouse_press(event)
            self.selected = not self.selected
            self.on_selection_callback(self.selected)

    def origin(self):
        return self.x, self.y

    def get_center_x(self) -> float:
        return self.x + (self.width() / 2)

    def get_center_y() -> float:
        pass
