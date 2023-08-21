import qtawesome as qta
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from nn_visualizer.application_context.application_context import (
    EditingDense,
    EditingDropout,
)
from nn_visualizer.application_context.editing_layers import EditingLayer
from nn_visualizer.ui_components.components_editors.editor_container import EditorContainer
from nn_visualizer.ui_components.components_editors.shared.component_name_container.component_name_container import ComponentNameContainer
from nn_visualizer.ui_components.components_editors.shared.units_number_change.units_number_change import UnitsNumberEditor
from nn_visualizer.ui_components.customized.double_slider.double_slider import DoubleSlider
from nn_visualizer.ui_components.customized.label.item_label import ItemLabel


class DropoutVisualizerEditor(EditorContainer):
    def __init__(self, editing_layer: EditingLayer):
        editor_specifics: list[QWidget] = self.items(editing_layer)
        EditorContainer.__init__(self, editing_layer, editor_specifics)

    def items(self, editing_dropout: EditingDropout) -> list[QLayoutItem]:
        dropout_slider = DoubleSlider(decimals=1, orientation=Qt.Orientation.Horizontal)
        dropout_slider.setMinimum(0)
        dropout_slider.setMaximum(1)
        dropout_slider.setValue(0)
        dropout_slider.setTracking(True)
        dropout_slider.valueChanged.connect(self.dropout_slider_value_changed)
   

        return [dropout_slider]

    def dropout_slider_value_changed(self, value: float):
        print('New dropout = {dropout}'.format(dropout=value))
