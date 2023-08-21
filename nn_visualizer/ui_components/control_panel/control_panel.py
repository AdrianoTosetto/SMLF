
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

from nn_visualizer.application_context.application_context import (
    ApplicationContext,
    ApplicationState,
)
from nn_visualizer.ui_components.components_editors.dense_visualizer_editor.dense_visualizer_editor import DenseVisualizerEditor
from nn_visualizer.ui_components.control_panel.buttons.step_buttons.next_step_button import NextStepButton
from nn_visualizer.ui_components.control_panel.buttons.step_buttons.play_full_button import PlayFullButton
from nn_visualizer.ui_components.control_panel.buttons.step_buttons.reset_button import ResetButton
from nn_visualizer.ui_components.control_panel.learning_rate_slider_container import LearningRateSliderContainer
from nn_visualizer.ui_components.customized.label.item_label import ItemLabel
from nn_visualizer.ui_components.losses.losses_panel import LossesPanel


class SliderContainer(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent,)
        self.epoch_slider = QSlider(parent, orientation=Qt.Orientation.Horizontal)
        self.epoch_slider.setObjectName('DoubleSlider')
        self.epoch_slider.setStyleSheet('#DoubleSlider { background-color: transparent; }')
        self.epoch_value_label = ItemLabel('Epoch: 0')
        self.epoch_slider.setRange(0, 100)
        self.epoch_slider.setValue(0)
        self.epoch_slider.setTracking(True)
        self.epoch_slider.valueChanged.connect(self.value_changed)

        self.root_layout = QVBoxLayout()
        self.root_layout.addWidget(self.epoch_value_label)
        self.root_layout.addWidget(self.epoch_slider)
        
        self.setLayout(self.root_layout)

    def value_changed(self):
        epoch_label_text = 'Epoch: {epoch}'.format(epoch=self.epoch_slider.value())
        self.epoch_value_label.setText(epoch_label_text)

class BatchSizeSliderContainer(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent,)
        self.batch_size_slider = QSlider(parent, orientation=Qt.Orientation.Horizontal)
        self.batch_size_slider.setObjectName('DoubleSlider')
        self.batch_size_slider.setStyleSheet('#DoubleSlider { background-color: transparent; }')
        self.epoch_value_label = ItemLabel("Batch Size: 0")
        self.batch_size_slider.setRange(0, 100)
        self.batch_size_slider.setValue(0)
        self.batch_size_slider.setTracking(True)
        self.batch_size_slider.valueChanged.connect(self.value_changed)

        self.root_layout = QVBoxLayout()
        self.root_layout.addWidget(self.epoch_value_label)
        self.root_layout.addWidget(self.batch_size_slider)
        
        self.setLayout(self.root_layout)

    def value_changed(self):
        epoch_label_text = 'Batch Size: {epoch}'.format(epoch=self.batch_size_slider.value())
        self.epoch_value_label.setText(epoch_label_text)

class MainButtons(QWidget):
    def __init__(self, application_context: ApplicationContext, parent=None):
        QWidget.__init__(self)
        self.reset_button = ResetButton(handler=None)
        self.next_step_button = NextStepButton(handler=application_context.next_step_training_click_handler)
        self.play_full_button = PlayFullButton(application_context=application_context)

        self.root_layout = QHBoxLayout()
        self.root_layout.addStretch()
        self.root_layout.addWidget(self.reset_button)
        self.root_layout.addWidget(self.play_full_button)
        self.root_layout.addWidget(self.next_step_button)
        self.root_layout.addStretch()
        self.setLayout(self.root_layout)

class ControlPanel(QWidget):
    def __init__(self, application_context: ApplicationContext, parent) -> None:
        QWidget.__init__(self)
        self.width = 450
        self.epochs_slider = SliderContainer()
        self.root_layout = QVBoxLayout()
        self.root_layout.setAlignment(Qt.AlignHCenter)
        self._build_layout(application_context, parent)

        application_context.pipeline.loss_change_listeners.append(self.listen_loss_change)

    def _build_layout(self, application_context: ApplicationContext, parent):
        self.setFixedWidth(self.width)
        self.root_layout.addWidget(ItemLabel('Training Panel', size='xs'))
        
        self.losses_panel = LossesPanel(parent=self)
        self.root_layout.addWidget(self.losses_panel)
        self.root_layout.addWidget(MainButtons(application_context, parent))
        self.root_layout.addWidget(LearningRateSliderContainer(application_context))
        self.root_layout.addWidget(SliderContainer(parent))
        self.root_layout.addWidget(BatchSizeSliderContainer(parent))
        self.root_layout.addStretch()
        self.setLayout(self.root_layout)
        self.setObjectName('ControlPanel')
        self.setStyleSheet('#ControlPanel { border: 1px solid #CCC; border-radius: 8px; background-color: white; }')
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

    def build_editing_model_layout(self):
        pass

    def clean_layout(self):
        layout = self.layout()
        item = layout.itemAt(0)
        layout.removeItem(item)
        for i in reversed(range(1, layout.count())): 
            widgetToRemove = layout.itemAt(i).widget()
            # remove it from the layout list
            layout.removeWidget(widgetToRemove)
            # remove it from the gui
            widgetToRemove.setParent(None)

    def paintEvent(self, a0: QPaintEvent) -> None:
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QPainter(self)
        QProxyStyle().drawPrimitive(QStyle.PE_Widget, opt, painter, self)
        QWidget.paintEvent(self, a0)

    def listen_loss_change(self, data):
        epochs, losses = data
        print(epochs, losses)
        self.losses_panel.update_train_loss_graph(epochs, losses)
