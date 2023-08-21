import PyQt5.QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QSlider, QVBoxLayout, QWidget

from nn_visualizer.application_context.application_context import ApplicationContext
from nn_visualizer.geometry.utils import orientation
from nn_visualizer.ui_components.customized.double_slider.double_slider import DoubleSlider
from nn_visualizer.ui_components.customized.label.item_label import ItemLabel


class LearningRateSliderContainer(QWidget):
    def __init__(self, application_context: ApplicationContext, parent=None):
        QWidget.__init__(self, parent,)
        self.epoch_slider = DoubleSlider(decimals=5, orientation=Qt.Orientation.Horizontal)
        self.epoch_value_label = ItemLabel('Learning Rate: 1.0')
        self.epoch_slider.setMinimum(0)
        self.epoch_slider.setMaximum(5)
        self.epoch_slider.setValue(1.)
        self.epoch_slider.setTracking(True)
        self.epoch_slider.valueChanged.connect(self.value_changed)

        self.root_layout = QVBoxLayout()
        self.root_layout.addWidget(self.epoch_value_label)
        self.root_layout.addWidget(self.epoch_slider)
        
        self.setLayout(self.root_layout)
        self.application_context = application_context

    def value_changed(self):
        lr = self.epoch_slider.value()
        epoch_label_text = 'Learning Rate: {lr}'.format(lr=lr)
        self.epoch_value_label.setText(epoch_label_text)
        self.application_context.set_learning_rate(lr)
