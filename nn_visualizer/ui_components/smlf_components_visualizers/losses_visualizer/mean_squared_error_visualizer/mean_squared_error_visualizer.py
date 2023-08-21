from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from api.losses.regression.mean_squared_loss import MeanSquareError
from nn_visualizer.geometry.point import Point
from nn_visualizer.geometry.polygon import Polygon
from nn_visualizer.ui_components.canvas_drawable import CanvasDrawable
from nn_visualizer.ui_components.smlf_components_visualizers.animations.current_pipeline_animation import CurrentPipelineAnimation
from nn_visualizer.ui_components.smlf_components_visualizers.layer_layout_data import LayerVisualizerLayoutData
from nn_visualizer.ui_components.smlf_components_visualizers.losses_visualizer.mean_squared_error_visualizer.loss_unit_visualizer_unit import LossVisualizerUnit


class MeanSquaredErrorVisualizer(CanvasDrawable):
    def __init__(self, layout_data: LayerVisualizerLayoutData, parent = None):
        self.parent = parent
        self.layout_data = layout_data
        self.x, self.y = layout_data.get_origin()
        self.polygon = Polygon.from_vertices(
            vertices=[
                Point(self.x, self.y),
                Point(self.x + self.layout_data.width(), self.y),
                Point(self.x + self.layout_data.width(), self.y + self.layout_data.height()),
                Point(self.x, self.y + self.layout_data.height()),
            ])

        self.animation = CurrentPipelineAnimation(parent=parent, polygon=self.polygon, update=self.update)
        self.animation.start()
        self.is_current = False
        self.selected = False
        self.name = 'MSE'

    def update(self):
        self.parent.update()

    def draw(self, painter: QPainter):
        brush = QBrush()
        brush.setColor(QColor('green'))
        painter.setPen(QColor("green"))
        brush.setStyle(Qt.SolidPattern)
        (x, y) = self.layout_data.get_origin()
        painter.drawText(QRectF(x, y, self.layout_data.width(), -25), Qt.AlignCenter, self.name)

        segments = self.animation.get_state()

        if self.is_current:
            for segment in segments:
                painter.drawLine(segment.to_QLine())

        for loss_visualizer_unit_origin in self.layout_data.get_units_origins():
            x, y = loss_visualizer_unit_origin
            LossVisualizerUnit(x, y).draw(painter)
