import numpy as np
import qtawesome as qta
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget

from nn_visualizer.application_context.application_context import EditingDense
from nn_visualizer.application_context.editing_layers import EditingLayer
from nn_visualizer.ui_components.components_editors.dense_visualizer_editor.activation_combo_box import ActivationComboBox
from nn_visualizer.ui_components.components_editors.editor_container import EditorContainer
from nn_visualizer.ui_components.components_editors.shared.component_name_container.component_name_container import ComponentNameContainer
from nn_visualizer.ui_components.components_editors.shared.units_number_change.units_number_change import UnitsNumberEditor
from nn_visualizer.ui_components.customized.label.item_label import ItemLabel


class DenseVisualizerEditor(EditorContainer):
    def __init__(self, editing_layer: EditingLayer):
        editor_specifics: list[QWidget] = self.items(editing_layer)
        EditorContainer.__init__(self, editing_layer, editor_specifics)

    def items(self, editing_dense: EditingDense) -> list[QLayoutItem]:
        
        groupbox = QGroupBox()
        hbox = QVBoxLayout()
        groupbox.setLayout(hbox)

        hbox.addWidget(ItemLabel('Initialization Params Algorithm'))
        radiobutton = QRadioButton("Random")
        radiobutton.setChecked(True)
        hbox.addWidget(radiobutton)

        radiobutton = QRadioButton("Normal")
        hbox.addWidget(radiobutton)

        activation_label = ItemLabel('Activation')
        activation_layout = QHBoxLayout()
        activation_layout.addWidget(activation_label)
        activation_layout.addWidget(ActivationComboBox(editing_dense))

        return [activation_layout, groupbox]
