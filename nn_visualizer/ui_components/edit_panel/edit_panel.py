
import sys
import typing

import qtawesome as qta
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtGui import QBrush, QPainter, QPen
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget

from api.layers.activation.softmax import Softmax
from nn_visualizer.application_context.application_context import (
    ApplicationContext,
    ApplicationState,
)
from nn_visualizer.ui_components.components_editors.dense_visualizer_editor.dense_visualizer_editor import DenseVisualizerEditor
from nn_visualizer.ui_components.control_panel.buttons.step_buttons.next_step_button import NextStepButton
from nn_visualizer.ui_components.control_panel.buttons.step_buttons.play_full_button import PlayFullButton
from nn_visualizer.ui_components.control_panel.buttons.step_buttons.reset_button import ResetButton
from nn_visualizer.ui_components.control_panel.learning_rate_slider_container import LearningRateSliderContainer
from nn_visualizer.ui_components.customized.drag_drop.smlf_drag_button import DragButton
from nn_visualizer.ui_components.customized.label.item_label import ItemLabel
from nn_visualizer.ui_components.edit_panel.buttons.add_batch_normalization_layer_drag_button import AddBatchNormalizationDragButton
from nn_visualizer.ui_components.edit_panel.buttons.add_dense_layer_drag_button import AddDenseDragButton
from nn_visualizer.ui_components.edit_panel.buttons.add_dropout_layer_drag_button import AddDropoutDragButton
from nn_visualizer.ui_components.edit_panel.buttons.add_relu_layer_drag_button import AddReLUDragButton
from nn_visualizer.ui_components.edit_panel.buttons.add_sigmoid_layer_drag_button import AddSigmoidDragButton
from nn_visualizer.ui_components.edit_panel.buttons.add_softmax_layer_drag_button import AddSoftmaxDragButton
from nn_visualizer.ui_components.edit_panel.loss_combobox import LossComboBox
from nn_visualizer.ui_components.edit_panel.optimizer_combox import OptmizerComboBox
from nn_visualizer.ui_components.losses.losses_panel import LossesPanel
from nn_visualizer.ui_components.theme.theme import Colors
from nn_visualizer.ui_components.theme.utils import json_style_to_css


class AddWidgetsPanel(QWidget):
    def __init__(self, parent) -> None:
        QWidget.__init__(self, parent)
        # self.setAttribute(Qt.WA_StyledBackground, True)
        # self.setObjectName('AddWidgetsPanel')
        # self.setStyleSheet('#AddWidgetsPanel {background-color: #af0C2D48; border-radius: 16px; }')

        effect = QGraphicsDropShadowEffect(
                offset=QPoint(3, 0), blurRadius=25, color=QColor("#AB0C2D48"))
    
        # self.setGraphicsEffect(effect)


        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(8)
        row_one_layout = QHBoxLayout()
        row_one_layout.addWidget(AddDenseDragButton(self), 3)
        row_one_layout.addWidget(AddDropoutDragButton(self), 2)
        
        row_two_layout = QHBoxLayout()
        row_two_layout.addWidget(AddBatchNormalizationDragButton(self))

        row_three_layout = QHBoxLayout()
        row_three_layout.addWidget(AddSigmoidDragButton(self))
        row_three_layout.addWidget(AddReLUDragButton(self))

        row_four_layout = QHBoxLayout()
        row_four_layout.addWidget(AddSoftmaxDragButton(self))

        root_layout.addLayout(row_one_layout)
        root_layout.addLayout(row_two_layout)
        root_layout.addLayout(row_three_layout)
        root_layout.addLayout(row_four_layout)
    
        root_layout.addWidget(ItemLabel('Loss Type'))
        root_layout.addWidget(LossComboBox(parent, on_changed_callback=lambda _: _))
        root_layout.addWidget(ItemLabel('Optimizer Type'))
        root_layout.addWidget(OptmizerComboBox(parent, on_changed_callback=lambda _: _))

        self.setLayout(root_layout)

class EditPanel(QWidget):
    def __init__(self) -> None:
        QWidget.__init__(self)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setObjectName('EditPanel')
        self.setStyleSheet('#EditPanel {background-color: #fafafa; border-radius: 8px; }')
        self.width = 400
        self.setFixedWidth(self.width)
        root_layout = QVBoxLayout()
        root_layout.addWidget(AddWidgetsPanel(self))
        root_layout.setAlignment(Qt.AlignTop)
        self.setLayout(root_layout)
        # self.setStyleSheet('background: green;')
