from __future__ import annotations
import functools
import os
import sys

import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from multipledispatch import dispatch

from api.batch_dense_normalization import BatchDenseNormalization
from api.layers.activation.relu import ReLU
from api.layers.activation.sigmoid import Sigmoid
from api.layers.activation.softmax import Softmax
from api.layers.dense import Dense
from api.layers.inv_dropout import InvDropout
from api.losses.binary_cross_entropy import BinaryCrossEntropy
from api.losses.regression.mean_squared_loss import MeanSquareError
from api.sequential.design_patterns.observer.observable import Event
from api.sequential.model import SequentialModel
from api.sequential.pipeline.decorators.dense_next_step_decorator import DenseNextStepDecorator
from api.sequential.pipeline.decorators.relu_decorator import ReLUNextStepDecorator
from api.sequential.pipeline.pipeline import Pipeline
from api.sequential.pipeline.pipeline_state import PipelineState
from iris_dataset import get
from nn_visualizer.application_context.application_context import (
    ApplicationContext,
    ApplicationState,
    EditingBatchNormalization,
    EditingDense,
    EditingDropout,
    EditingInput,
)
from nn_visualizer.application_context.editing_layers import (
    EditingBinaryCrossEntropy,
    EditingDropout,
    EditingLayer,
    EditingReLU,
    EditingSigmoid,
    EditingSoftmax,
)
from nn_visualizer.external.default_file_adapter import DefaultFileAdapter
from nn_visualizer.infra.file_manager.file_manager import File
from nn_visualizer.layout_data import LayoutData
from nn_visualizer.ui_components.canvas_widget import CanvasWidget
from nn_visualizer.ui_components.components_editors.batch_norm_editor.batch_normalization_editor import BatchNormalizationEditor
from nn_visualizer.ui_components.components_editors.dense_visualizer_editor.dense_visualizer_editor import DenseVisualizerEditor
from nn_visualizer.ui_components.components_editors.dropout_visualizer_editor.dropout_visualizer_editor import DropoutVisualizerEditor
from nn_visualizer.ui_components.components_editors.input_visualizer_editor.input_visualizer_editor import InputVisualizerEditor
from nn_visualizer.ui_components.components_editors.relu_editor.relu_visualizer_editor import ReLUVisualizerEditor
from nn_visualizer.ui_components.components_editors.sigmoid_editor.sigmoid_visualizer_editor import SigmoidVisualizerEditor
from nn_visualizer.ui_components.components_editors.softmax_visualizer_editor.softmax_visualizer_editor import SoftmaxVisualizerEditor
from nn_visualizer.ui_components.console.console import Console
from nn_visualizer.ui_components.control_panel.control_panel import ControlPanel
from nn_visualizer.ui_components.derivatives_drawer.derivatives_drawer import DerivativesDrawer
from nn_visualizer.ui_components.edit_panel.edit_panel import EditPanel
from nn_visualizer.ui_components.layout_data.default_batch_normalization_layout_data import DefaultBatchNormalizationVisualizerLayoutData
from nn_visualizer.ui_components.layout_data.default_dense_visualizer_layout_data import DefaultDenseVisualizerLayoutData
from nn_visualizer.ui_components.layout_data.default_dropout_visualizer_layout_data import DefaultDropoutVisualizerLayoutData
from nn_visualizer.ui_components.layout_data.default_input_layout_data import DefaultInputVisualizerLayoutData
from nn_visualizer.ui_components.layout_data.default_relu_visualizer_layout_data import DefaultReLUVisualizerLayoutData
from nn_visualizer.ui_components.layout_data.default_sigmoid_layout_data import DefaultSigmoidVisualizerLayoutData
from nn_visualizer.ui_components.layout_data.default_softmax_visualizer_layout_data import DefaultSoftmaxVisualizerLayoutData
from nn_visualizer.ui_components.menubar.main_window_menu_bar import MainWindowMenuBar
from nn_visualizer.ui_components.smlf_components_visualizers.batch_normalization_visualizer.batch_normalization_unit_visualizer import BatchNormalizationUnitVisualizer
from nn_visualizer.ui_components.smlf_components_visualizers.batch_normalization_visualizer.batch_normalization_visualizer import BatchNormalizationVisualizer
from nn_visualizer.ui_components.smlf_components_visualizers.dense_visualizer.visualizer_dense_layer import DenseVisualizer
from nn_visualizer.ui_components.smlf_components_visualizers.dropout_visualizer.dropout_visualizer import DropoutVisualizer
from nn_visualizer.ui_components.smlf_components_visualizers.input_visualizer.input_visualizer import InputVisualizer
from nn_visualizer.ui_components.smlf_components_visualizers.layer_layout_data import LayerVisualizerLayoutData
from nn_visualizer.ui_components.smlf_components_visualizers.layer_unit_visualizer import Showing
from nn_visualizer.ui_components.smlf_components_visualizers.losses_visualizer.binary_cross_entropy_visualizer.binary_cross_entropy_visualizer import BinaryCrossEntropyVisualizer
from nn_visualizer.ui_components.smlf_components_visualizers.losses_visualizer.mean_squared_error_visualizer.mean_squared_error_visualizer import MeanSquaredErrorVisualizer
from nn_visualizer.ui_components.smlf_components_visualizers.relu_visualizer.relu_visualizer import ReLUVisualizer
from nn_visualizer.ui_components.smlf_components_visualizers.sigmoid_visualizer.sigmoid_visualizer import SigmoidVisualizer
from nn_visualizer.ui_components.smlf_components_visualizers.strategies import line_color_from_derivative


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



# SPACE_BETWEEN_NODES = 60
# SPACE_BETWEEN_LAYERS = 250
START_DRAWING_X = 20
START_DRAWING_Y = 20

CANVAS_WIDTH = 1400
CANVAS_HEIGHT = 500

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet('background: #e0e0e0;')

        self.menubar = MainWindowMenuBar(parent=self,
                open_file_callback=self.open_file_callback,
                switch_view_mode_callback=self.switch_view_mode_callback,
        )

        self.layout_data = LayoutData(self, self.on_selection_callback, self.on_node_selection_callback)
        pipeline = Pipeline([], next_callback=self.update_pipeline_callback)
        pipeline.loss_change_listeners = [self.loss_listener]
        self.current_x = 0

        self.application_context = ApplicationContext(
                pipeline,
                add_layer_callback=self.add_layer_callback,
                update_layer_callback=self.layer_update_callback,
            )

        self.canvas = CanvasWidget(self.application_context, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, parent=self)

        self.control_panel = ControlPanel(self.application_context, self)
        self.editing_panel = EditPanel()
        self.layout_data.add_input(self.application_context.editing_layers[0])
        self.layout_data.add_loss_visualizer(self.application_context.editing_loss)
        self.canvas.set_drawables(self.layout_data.layers)
        self.console = Console(self.application_context, self.handle_load_callback)
        self.application_context.pipeline.add(self.console.pipeline_update_handler)
        self.application_context.pipeline.add(self.pipeline_update_handler)

        self.width = 1800
        self.height = CANVAS_HEIGHT + 300

        self._build_layout()

        self.show()

    def handle_load_callback(self, dataset, targets, validation_dataset, validation_targets):
        layers = self.compile_editing_model()
        bce_units = layers[-1].units
        self.dataset = dataset
        self.targets = targets
        self.validation_dataset = validation_dataset
        self.validation_targets = validation_targets

    def pipeline_update_handler(self, event: Event):
        for layer in self.layout_data.layers:
            layer.layout_data.set_pipeline_state(self.application_context.pipeline.state)

    def loss_listener(self, epochs, losses, validation_losses):
        self.control_panel.losses_panel.update_train_loss_graph(epochs, losses, validation_losses)
        self.update()

    def layer_update_callback(self, index: int, editing_layer: EditingLayer):
        editing_layers = self.application_context.editing_layers
        selection_statuses = list(map(lambda visualizer: visualizer.selected, self.layout_data.layers))
        self.layout_data.layers = []
        self.layout_data.add_input(editing_layers[0])

        for index, layer in enumerate(editing_layers[1:], start=1):
            layer = self.layout_data.visualizer_from_editing(editing_layers[index], index)
            self.layout_data.layers.append(layer)

        for index, layer in enumerate(self.layout_data.layers):
            self.layout_data.set_visualizer_layer(layer, index)
            layer = selection_statuses[index]

        for layer, selection_status in zip(self.layout_data.layers, selection_statuses):
            layer.selected = selection_status

        self.layout_data.add_loss_visualizer(self.application_context.editing_loss)
        self.canvas.set_drawables(self.layout_data.layers)
        self.update()

    def switch_view_mode_callback(self, state: ApplicationState):
        if state == self.application_context.state:
            return

        if state == ApplicationState.IDLE_TRAINING:
            layers = self.compile_editing_model()
            bce_units = layers[-1].units
            self.application_context.pipeline.set_pipeline(layers + [BinaryCrossEntropy(bce_units)], self.dataset, self.loss_listener)
            self.application_context.pipeline.set_targets(self.targets)
            self.application_context.pipeline.set_validation_inputs(self.validation_dataset)
            self.application_context.pipeline.set_validation_targets(self.validation_targets)
            self._swtich_panel(self.control_panel)

        elif state == ApplicationState.EDITING:
            self._swtich_panel(self.editing_panel)

        self.application_context.set_state(state)

    def open_file_callback(self, path):
        file: File = DefaultFileAdapter(path, File.OpenPolicy.write)
        file.open()
        jsonstr = file.read()
        model = SequentialModel.from_string_json(jsonstr)
        layers = model.layers + [model.loss]
        layout_data = LayoutData(self, layers)
        pipeline = Pipeline(layers, next_callback=self.update_callback)
        self.application_context = ApplicationContext(pipeline, layout_data)

        self.canvas.set_drawables(self.application_context.layout_data.layers)
        self.update()

    def update_pipeline_callback(self, index):
        layer = self.application_context.pipeline.layers[index]
        for i in range(len(self.layout_data.layers)):
            self.layout_data.layers[i].is_current = False
        self.layout_data.layers[index+1].is_current = True

        if self.application_context.pipeline.state == PipelineState.BACKWARD and not layer.last:
            next_layer_index = index
            next_layer = self.application_context.pipeline.layers[next_layer_index+1]

            derivatives = next_layer.mean_output_derivatives_wrt_inputs

            layout_data: LayerVisualizerLayoutData = self.layout_data.layers[index+1].layout_data
            layout_data.set_backpropagation_derivatives(derivatives)

            if self.selected_node_info is not None:
                (layer_index, node_index) = self.selected_node_info
                if index + 1 == layer_index:
                    shape_len = len(derivatives.shape)
                    colors = None

                    if shape_len == 1:
                        r,g,b = line_color_from_derivative(np.abs(derivatives[node_index]))
                        colors = [QColor(r, g, b)]
                    else:
                        def to_QColor(rgb):
                            r,g,b = rgb
                            return QColor(r,g,b)

                        colors = list(map(lambda derivative: to_QColor(line_color_from_derivative(np.abs(derivative))), derivatives[:,node_index]))

                    self.canvas.legend.derivatives_colors = colors

        self.update()

    def _build_layout(self):
        root_layout = QVBoxLayout()
        canvas_console_layout = QVBoxLayout()
        panel_layout = QHBoxLayout()

        canvas_console_layout.addWidget(self.canvas, 1)
        canvas_console_layout.addWidget(self.console)

        panel_layout.addLayout(canvas_console_layout)
        if self.application_context.state == ApplicationState.EDITING:
            panel_layout.addWidget(self.editing_panel)
        else:
            panel_layout.addWidget(self.control_panel)

        root_layout.addWidget(self.menubar)
        root_layout.addLayout(panel_layout)

        self.setLayout(root_layout)

        self.setGeometry(0, 0, self.width, self.height)

    def _swtich_panel(self, replacement: QWidget):
        layout = self.layout().itemAt(1).layout()
        layout.addWidget(replacement)
        widget = layout.itemAt(1).widget()
        layout.removeWidget(widget)
        widget.setParent(None)

    def set_dense_visualizer_editor(self, index: int):
        layout: QVBoxLayout = self.editing_panel.layout()
        editing_layer = self.application_context.editing_layers[index]
        layout.addWidget(DenseVisualizerEditor(editing_layer))
        layout.addStretch()

    def set_dropout_visualizer_editor(self, index):
        layout: QVBoxLayout = self.editing_panel.layout()
        editing_layer = self.application_context.editing_layers[index]
        layout.addWidget(DropoutVisualizerEditor(editing_layer))
        layout.addStretch()

    def remove_visualizer_editor(self):
        layout = self.editing_panel.layout()
        editor = layout.itemAt(1).widget()
        stretch = layout.itemAt(2)

        if editor:
            layout.removeWidget(editor)
            layout.removeItem(stretch)
            editor.setParent(None)
    
    def update(self):
        self.canvas.set_drawables(self.layout_data.layers)

    def visualizers_from_editing(self):
        self.current_x = 30

        for (index, layer) in enumerate(self.application_context.editing_layers):
            visualizer = self.layout_data.visualizer_from_editing(layer, index)
            self.layout_data.current_x += visualizer.layout_data.width()
            self.layout_data.add_visualizer_layer(visualizer, index)

    @dispatch(EditingDense)
    def set_editor(self, editing_layer: EditingDense):
        layout: QVBoxLayout = self.editing_panel.layout()
        layout.addWidget(DenseVisualizerEditor(editing_layer, ))
        layout.addStretch()

    @dispatch(EditingReLU)
    def set_editor(self, editing_layer: EditingReLU):
        layout: QVBoxLayout = self.editing_panel.layout()
        layout.addWidget(ReLUVisualizerEditor(editing_layer, ))
        layout.addStretch()

    @dispatch(EditingSigmoid)
    def set_editor(self, editing_layer: EditingSigmoid):
        layout: QVBoxLayout = self.editing_panel.layout()
        layout.addWidget(SigmoidVisualizerEditor(editing_layer, ))
        layout.addStretch()

    @dispatch(EditingDropout)
    def set_editor(self, editing_layer: EditingDropout):
        layout: QVBoxLayout = self.editing_panel.layout()
        layout.addWidget(DropoutVisualizerEditor(editing_layer))
        layout.addStretch()

    @dispatch(EditingBatchNormalization)
    def set_editor(self, editing_layer: EditingBatchNormalization):
        layout: QVBoxLayout = self.editing_panel.layout()
        layout.addWidget(BatchNormalizationEditor(editing_layer))
        layout.addStretch()

    @dispatch(EditingInput)
    def set_editor(self, editing_input: EditingInput):
        layout: QVBoxLayout = self.editing_panel.layout()
        layout.addWidget(InputVisualizerEditor(editing_input))
        layout.addStretch()

    @dispatch(EditingSoftmax)
    def set_editor(self, editing_softmax: EditingSoftmax):
        layout: QVBoxLayout = self.editing_panel.layout()
        layout.addWidget(SoftmaxVisualizerEditor(editing_softmax))
        layout.addStretch()

    def on_selection_callback(self, editing_index: int, selected):
        layer = self.application_context.editing_layers[editing_index]

        if selected:
            editing_layer = self.application_context.editing_layers[editing_index]
            self.set_editor(editing_layer)
        else:
            self.remove_visualizer_editor()

    def on_node_selection_callback(self, layer_index: int, node_index: int):
        self.selected_node_info = (layer_index, node_index)
        dependencies = self.application_context.trace_dependency_path(layer_index, node_index)
        self.layout_data.layers[layer_index].drawable_units[node_index].showing_info = Showing.SHOWING_BACKWARD_CONNECTIONS_IN_PATH

        for dlayer_index, layer_dependencies in dependencies:
            for index in layer_dependencies:
                self.layout_data.layers[dlayer_index].drawable_units[index].showing_info = Showing.SHOWING_BACKWARD_CONNECTIONS_IN_PATH


        next_layer = self.application_context.editing_layers[layer_index+1]
        next_layer_node_dependents = next_layer.get_nodes_indexes_linked_with_previous_node(node_index)

        derivatives_visualizer = DerivativesDrawer(
            layer_index,
            node_index,
            next_layer_node_dependents,
            list(map(lambda node_index: QColor(0, 0, 0), next_layer_node_dependents))
        )

        self.canvas.set_legend(derivatives_visualizer)

    def compile_editing_model(self):
        model_layers = list(map(self.layer_from_editing_model, self.application_context.editing_layers[1:]))
        return model_layers

    @dispatch(EditingDense)
    def layer_from_editing_model(self, editing_dense: EditingDense) -> Dense:
        units = editing_dense.units
        ninputs = editing_dense.ninputs
        init_algorithm = editing_dense.params_init_algorithm
    
        return Dense(ninputs=ninputs, noutputs=units, init_algorithm=init_algorithm)

    @dispatch(EditingDropout)
    def layer_from_editing_model(self, editing_dropout: EditingDropout):
        units = editing_dropout.units
        dropout_rate = editing_dropout.dropout_rate
        return InvDropout(units, dropout_rate)

    @dispatch(EditingSigmoid)
    def layer_from_editing_model(self, editing_sigmoid: EditingSigmoid):
        units = editing_sigmoid.units
        return Sigmoid(units)

    @dispatch(EditingReLU)
    def layer_from_editing_model(self, editing_relu: EditingReLU):
        units = editing_relu.units
        return ReLU(units)

    @dispatch(EditingSoftmax)
    def layer_from_editing_model(self, editing_softmax: EditingSoftmax):
        units = editing_softmax.units
        return Softmax(units)

    @dispatch(EditingBatchNormalization)
    def layer_from_editing_model(self, editing_batch_normalization: EditingBatchNormalization):
        units = editing_batch_normalization.units
        return BatchDenseNormalization(units)

    def add_layer_callback(self, editing_layer: EditingLayer, index: int):
        self.layout_data.add_layer(editing_layer, index)
        self.canvas.set_drawables(self.layout_data.layers)

App = QApplication(sys.argv)

window = Window()
sys.exit(App.exec())
