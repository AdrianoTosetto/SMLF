import PyQt5.QtWidgets
import numpy as np
import qtawesome as qta
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from nn_visualizer.application_context.editing_layers import EditingLayer
from nn_visualizer.ui_components.components_editors.shared.component_name_container.component_name_container import ComponentNameContainer
from nn_visualizer.ui_components.components_editors.shared.units_number_change.units_number_change import UnitsNumberEditor


class EditorContainer(QWidget):
    def __init__(self, editing_layer: EditingLayer, editor_specifics: list[QWidget] = []):
        QWidget.__init__(self)
        self.editing_layer = editing_layer
        self.setObjectName('DenseVisualizerEditor')
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.build_layout(editor_specifics)
        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Maximum
        )
        # self.setStyleSheet('#DenseVisualizerEditor {background-color: red; border-radius: 4px; }')
        effect = QGraphicsDropShadowEffect(
            offset=QPoint(3, 3), blurRadius=25, color=QColor("#e0e0e0"))
    
        self.setGraphicsEffect(effect)
        self.editor_specifics = editor_specifics

    def build_layout(self, editor_specifics: list[QGraphicsLayoutItem]):
        root_layout = QVBoxLayout()
        units_number_layout = QHBoxLayout()
        units_number_layout.addWidget(ComponentNameContainer(self, self.layer_name_changed, default_text=self.editing_layer.name))
        units_number_layout.addWidget(UnitsNumberEditor(self.editing_layer))

        root_layout.addLayout(units_number_layout)
        for widget in editor_specifics:
            if isinstance(widget, QWidget):
                root_layout.addWidget(widget)
            if isinstance(widget, QLayout):
                root_layout.addLayout(widget)
    
        self.setLayout(root_layout)

    def layer_name_changed(self, text: str):
        self.editing_layer.name = text
        print(self.editing_layer.name)
        self.editing_layer.update()
