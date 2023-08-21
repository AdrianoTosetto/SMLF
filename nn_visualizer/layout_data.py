import functools
from collections import deque
from copy import deepcopy

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtGui import QBrush, QPainter, QPen
from PyQt5.QtWidgets import *
from multipledispatch import dispatch

from api.layers.activation.relu import ReLU
from api.layers.dense import Dense
from api.losses.regression.mean_squared_loss import MeanSquareError
from nn_visualizer.application_context.editing_layers import (
    EditingBatchNormalization,
    EditingBinaryCrossEntropy,
    EditingDense,
    EditingDropout,
    EditingInput,
    EditingLayer,
    EditingMeanSquaredError,
    EditingReLU,
    EditingSigmoid,
    EditingSoftmax,
)
from nn_visualizer.ui_components.layout_data.default_batch_normalization_layout_data import DefaultBatchNormalizationVisualizerLayoutData
from nn_visualizer.ui_components.layout_data.default_binary_crossentropy_visualizer import DefaultBinaryCrossEntropyLayoutData
from nn_visualizer.ui_components.layout_data.default_dense_visualizer_layout_data import DefaultDenseVisualizerLayoutData
from nn_visualizer.ui_components.layout_data.default_dropout_visualizer_layout_data import DefaultDropoutVisualizerLayoutData
from nn_visualizer.ui_components.layout_data.default_input_layout_data import DefaultInputVisualizerLayoutData
from nn_visualizer.ui_components.layout_data.default_mean_squared_loss_visualizer_data import DefaultMeanSquaredLossVisualizerLayoutData
from nn_visualizer.ui_components.layout_data.default_relu_visualizer_layout_data import DefaultReLUVisualizerLayoutData
from nn_visualizer.ui_components.layout_data.default_sigmoid_layout_data import DefaultSigmoidVisualizerLayoutData
from nn_visualizer.ui_components.layout_data.default_softmax_visualizer_layout_data import DefaultSoftmaxVisualizerLayoutData
from nn_visualizer.ui_components.smlf_components_visualizers.batch_normalization_visualizer.batch_normalization_unit_visualizer import BatchNormalizationUnitVisualizer
from nn_visualizer.ui_components.smlf_components_visualizers.batch_normalization_visualizer.batch_normalization_visualizer import BatchNormalizationVisualizer
from nn_visualizer.ui_components.smlf_components_visualizers.dense_visualizer.visualizer_dense_layer import DenseVisualizer
from nn_visualizer.ui_components.smlf_components_visualizers.dropout_visualizer.dropout_visualizer import DropoutVisualizer
from nn_visualizer.ui_components.smlf_components_visualizers.input_visualizer.input_visualizer import InputVisualizer
from nn_visualizer.ui_components.smlf_components_visualizers.layer_layout_data import LayerVisualizerLayoutData
from nn_visualizer.ui_components.smlf_components_visualizers.losses_visualizer.binary_cross_entropy_visualizer.binary_cross_entropy_visualizer import BinaryCrossEntropyVisualizer
from nn_visualizer.ui_components.smlf_components_visualizers.losses_visualizer.mean_squared_error_visualizer.mean_squared_error_visualizer import MeanSquaredErrorVisualizer
from nn_visualizer.ui_components.smlf_components_visualizers.relu_visualizer.relu_visualizer import ReLUVisualizer
from nn_visualizer.ui_components.smlf_components_visualizers.sigmoid_visualizer.sigmoid_visualizer import SigmoidVisualizer
from nn_visualizer.ui_components.smlf_components_visualizers.strategies import (
    draw_fully_connected_backward_connections,
    draw_fully_connected_forward_connections,
    draw_single_connected_backward_connection,
    draw_single_connected_forward_connection,
)


class LayoutData():
    def __init__(self, parent, on_selection_callback, on_node_selection_callback):
        self.on_selection_callback = on_selection_callback
        self.on_node_selection_callback = on_node_selection_callback
        self.current_x = 20
        self.parent = parent
        self.x_origin = 20 
        self.layers = []

    def add_input(self, editing_input: EditingInput):
        def generate_selection_callback(index: int):
            def selection_callback(selected: bool):
                self.on_selection_callback(index, selected)

            return selection_callback
        input_visualizer = self.visualizer_from_editing(editing_input, 0)
        self.layers.append(input_visualizer)

    def add_loss_visualizer(self, editing_input: EditingBinaryCrossEntropy):
        def generate_selection_callback(index: int):
            def selection_callback(selected: bool):
                self.on_selection_callback(index, selected)

            return selection_callback
        loss_visualizer = self.visualizer_from_editing(editing_input, len(self.layers))
        self.layers.append(loss_visualizer)
        self.set_visualizer_layer(loss_visualizer, len(self.layers) - 1)

    def add_layer(self, editing_layer: EditingLayer, index: int):
        visualizer = self.visualizer_from_editing(editing_layer, index)

        self.set_visualizer_layer(visualizer, index)
        self.add_loss_visualizer(EditingBinaryCrossEntropy(units=3))

    def on_selection_callback(self, editing_layer: EditingLayer, selected: bool):
        self.selection_callback(editing_layer, selected)

    def update_layer(self, index: int, editing_layer: EditingLayer):
        selected = self.layers[index].selected
        visualizer = self.visualizer_from_editing(editing_layer, index)
        visualizer.selected = selected
        self.set_visualizer_layer(visualizer, index)

    def calculate_current_x(self, index):
        if index == 0:
            return 20

        from functools import reduce
        return reduce(lambda a, b: a + b, map(lambda layer: layer.layout_data.width(), self.layers[0:index]), 20)

    @dispatch(EditingDense, int)
    def visualizer_from_editing(self, editing_dense: EditingDense, index: int):
        units = editing_dense.units
        def generate_selection_callback(index: int):
            def selection_callback(selected: bool):
                self.on_selection_callback(index, selected)

            return selection_callback

        def generate_on_node_selection_callback(index: int):
            def node_selection_callback(node_index: int):
                self.on_node_selection_callback(index, node_index)
    
            return node_selection_callback

        layout_data = DefaultDenseVisualizerLayoutData(units, index, self.calculate_current_x(index))


        return DenseVisualizer(
            layout_data,
            parent=self.parent,
            on_selection_callback=generate_selection_callback(index),
            on_node_selection_callback=generate_on_node_selection_callback(index),
            name=editing_dense.name,
        )

    @dispatch(EditingDropout, int)
    def visualizer_from_editing(self, editing_dropout: EditingDropout, index: int):
        units = editing_dropout.units
        def generate_selection_callback(index: int):
            def selection_callback(selected: bool):
                self.on_selection_callback(index, selected)

            return selection_callback

        layout_data = DefaultDropoutVisualizerLayoutData(units, index, self.calculate_current_x(index))

        return DropoutVisualizer(
            units,
            layout_data,
            parent=self.parent,
            on_selection_callback=generate_selection_callback(index),
            name=editing_dropout.name)

    @dispatch(EditingBatchNormalization, int)
    def visualizer_from_editing(self, editing_layer: EditingBatchNormalization, index: int):
        units = editing_layer.units
        def generate_selection_callback(index: int):
            def selection_callback(selected: bool):
                self.on_selection_callback(index, selected)

            return selection_callback

        def generate_on_node_selection_callback(index: int):
            def node_selection_callback(node_index: int):
                self.on_node_selection_callback(index, node_index)

            return node_selection_callback


        layout_data = DefaultBatchNormalizationVisualizerLayoutData(units, index, self.calculate_current_x(index))

        return BatchNormalizationVisualizer(layout_data,
                parent=self.parent,
                name=editing_layer.name,
                on_selection_callback=generate_selection_callback(index),
                on_node_selection_callback=generate_on_node_selection_callback(index)
                
            )

    @dispatch(EditingInput, int)
    def visualizer_from_editing(self, editing_input: EditingInput, index: int):
        def generate_selection_callback(index: int):
            def selection_callback(selected: bool):
                self.on_selection_callback(index, selected)

            return selection_callback

        layout_data = DefaultInputVisualizerLayoutData(editing_input.units, index, self.calculate_current_x(index))

        return InputVisualizer(
                    editing_input.units,
                    layout_data,
                    parent=self.parent,
                    on_selection_callback=generate_selection_callback(index)
                    )

    @dispatch(EditingBinaryCrossEntropy, int)
    def visualizer_from_editing(self, editing_binary_cross_entropy: EditingBinaryCrossEntropy, index: int):
        def generate_selection_callback(index: int):
            def selection_callback(selected: bool):
                self.on_selection_callback(index, selected)

            return selection_callback

        layout_data = DefaultBinaryCrossEntropyLayoutData(editing_binary_cross_entropy.units, index, self.calculate_current_x(index))

        return BinaryCrossEntropyVisualizer(
            editing_binary_cross_entropy.units,
            layout_data, parent=self.parent,
            on_selection_callback=generate_selection_callback(index))

    @dispatch(EditingMeanSquaredError, int)
    def visualizer_from_editing(self, editing_mean_squared_error: EditingMeanSquaredError, index: int):
        def generate_selection_callback(index: int):
            def selection_callback(selected: bool):
                self.on_selection_callback(index, selected)

            return selection_callback

        layout_data = DefaultMeanSquaredLossVisualizerLayoutData(editing_mean_squared_error.units, index, self.calculate_current_x(index))

        return MeanSquaredErrorVisualizer(
            layout_data,
            parent=self.parent)

    @dispatch(EditingSigmoid, int)
    def visualizer_from_editing(self, editing_sigmoid: EditingSigmoid, index: int):
        def generate_selection_callback(index: int):
            def selection_callback(selected: bool):
                self.on_selection_callback(index, selected)

            return selection_callback

        def generate_on_node_selection_callback(index: int):
            def node_selection_callback(node_index: int):
                self.on_node_selection_callback(index, node_index)
    
            return node_selection_callback

        layout_data = DefaultSigmoidVisualizerLayoutData(editing_sigmoid.units, index, self.calculate_current_x(index))

        return SigmoidVisualizer(
                    layout_data,
                    parent=self.parent,
                    on_selection_callback=generate_selection_callback(index),
                    on_node_selection_callback=generate_on_node_selection_callback(index),
                    name=editing_sigmoid.name)

    @dispatch(EditingReLU, int)
    def visualizer_from_editing(self, editing_relu: EditingReLU, index: int):
        def generate_selection_callback(index: int):
            def selection_callback(selected: bool):
                self.on_selection_callback(index, selected)

            return selection_callback

        def generate_on_node_selection_callback(index: int):
            def node_selection_callback(node_index: int):
                self.on_node_selection_callback(index, node_index)
    
            return node_selection_callback

        layout_data = DefaultReLUVisualizerLayoutData(editing_relu.units, index, self.calculate_current_x(index))

        return ReLUVisualizer(
                editing_relu.units,
                layout_data,
                parent=self.parent,
                on_selection_callback=generate_selection_callback(index),
                on_node_selection_callback=generate_on_node_selection_callback(index),
                name=editing_relu.name
            )

    @dispatch(EditingSoftmax, int)
    def visualizer_from_editing(self, editing_relu: EditingSoftmax, index: int):
        def generate_selection_callback(index: int):
            def selection_callback(selected: bool):
                self.on_selection_callback(index, selected)

            return selection_callback

        layout_data = DefaultSoftmaxVisualizerLayoutData(editing_relu.units, index, self.calculate_current_x(index))

        return ReLUVisualizer(editing_relu.units, layout_data,
                parent=self.parent, on_selection_callback=generate_selection_callback(index), name=editing_relu.name)

    @dispatch(DenseVisualizer, int)
    def set_visualizer_layer(self, dense_visualizer: DenseVisualizer, index: int):
        previous_layer_layout_data: LayerVisualizerLayoutData = self.layers[index-1].layout_data
        dense_unit_radius = dense_visualizer.layout_data.get_unit_radius()
        previous_layer_unit_radius = previous_layer_layout_data.get_unit_radius()

        dense_units_origins = dense_visualizer.layout_data.get_units_origins()
        dense_units_origins = list(map(lambda origin: self.translate_x(origin, -dense_unit_radius), dense_units_origins))
        previous_layer_units_origins = previous_layer_layout_data.get_units_origins()
        previous_layer_units_origins = list(map(lambda origin: self.translate_x(origin, previous_layer_unit_radius), previous_layer_units_origins))

        previous_layer_layout_data.set_output_units_origins(dense_units_origins)
        dense_visualizer.layout_data.set_draw_forward_connections_strategy(draw_fully_connected_forward_connections)
        previous_layer_layout_data.set_draw_backward_connections_strategy(draw_fully_connected_backward_connections)
        dense_visualizer.layout_data.set_input_units_origins(previous_layer_units_origins)

        self.layers[index] = dense_visualizer

    @dispatch(SigmoidVisualizer, int)
    def set_visualizer_layer(self, sigmoid_visualizer: SigmoidVisualizer, index: int):
        previous_layer_layout_data: LayerVisualizerLayoutData = self.layers[index-1].layout_data
        previous_layer_unit_radius = previous_layer_layout_data.get_unit_radius()
        sigmoid_unit_radius = sigmoid_visualizer.layout_data.get_unit_radius()
    
        sigmoid_units_origins = sigmoid_visualizer.layout_data.get_units_origins()
        sigmoid_units_origins = list(map(lambda origin: self.translate_x(origin, -sigmoid_unit_radius), sigmoid_units_origins))
        previous_layer_units_origins = previous_layer_layout_data.get_units_origins()
        previous_layer_units_origins = list(map(lambda origin: self.translate_x(origin, previous_layer_unit_radius), previous_layer_units_origins))

        previous_layer_layout_data.set_output_units_origins(sigmoid_units_origins)
        previous_layer_layout_data.set_draw_backward_connections_strategy(draw_single_connected_backward_connection)

        sigmoid_visualizer.layout_data.set_draw_forward_connections_strategy(draw_single_connected_forward_connection)
        sigmoid_visualizer.layout_data.set_input_units_origins(previous_layer_units_origins)

        self.layers[index] = sigmoid_visualizer

    @dispatch(ReLUVisualizer, int)
    def set_visualizer_layer(self, relu_visualizer: ReLUVisualizer, index: int):
        previous_layer_layout_data: LayerVisualizerLayoutData = self.layers[index-1].layout_data
        previous_layer_unit_radius = previous_layer_layout_data.get_unit_radius()
        relu_unit_radius = relu_visualizer.layout_data.get_unit_radius()

        relu_units_origins = relu_visualizer.layout_data.get_units_origins()
        relu_units_origins = list(map(lambda origin: self.translate_x(origin, -relu_unit_radius), relu_units_origins))
        previous_layer_units_origins = previous_layer_layout_data.get_units_origins()
        previous_layer_units_origins = list(map(lambda origin: self.translate_x(origin, previous_layer_unit_radius), previous_layer_units_origins))

        previous_layer_layout_data.set_output_units_origins(relu_units_origins)
        previous_layer_layout_data.set_draw_backward_connections_strategy(draw_single_connected_backward_connection)

        relu_visualizer.layout_data.set_draw_forward_connections_strategy(draw_single_connected_forward_connection)
        relu_visualizer.layout_data.set_input_units_origins(previous_layer_units_origins)

        self.layers[index] = relu_visualizer

    @dispatch(DropoutVisualizer, int)
    def set_visualizer_layer(self, dropout_visualizer: DropoutVisualizer, index: int):
        previous_layer_layout_data: LayerVisualizerLayoutData = self.layers[index-1].layout_data
        previous_layer_unit_radius = previous_layer_layout_data.get_unit_radius()

        dense_units_origins = dropout_visualizer.layout_data.get_units_origins()
        previous_layer_units_origins = previous_layer_layout_data.get_units_origins()
        previous_layer_units_origins = list(map(lambda origin: self.translate_x(origin, previous_layer_unit_radius), previous_layer_units_origins))

        previous_layer_layout_data.set_output_units_origins(dense_units_origins)
        previous_layer_layout_data.set_draw_backward_connections_strategy(draw_single_connected_backward_connection)
        previous_layer_layout_data.set_draw_forward_connections_strategy(draw_single_connected_forward_connection)
        dropout_visualizer.layout_data.set_input_units_origins(previous_layer_units_origins)

        self.layers[index] = dropout_visualizer

    @dispatch(BatchNormalizationVisualizer, int)
    def set_visualizer_layer(self, batch_normalization_visualizer: BatchNormalizationUnitVisualizer, index: int):
        previous_layer_layout_data: LayerVisualizerLayoutData = self.layers[index-1].layout_data

        dense_units_origins = batch_normalization_visualizer.layout_data.get_units_origins()
        previous_layer_units_origins = previous_layer_layout_data.get_units_origins()

        previous_layer_layout_data.set_draw_forward_connections_strategy(draw_single_connected_forward_connection)
        previous_layer_layout_data.set_draw_backward_connections_strategy(draw_single_connected_backward_connection)
        previous_layer_layout_data.set_output_units_origins(dense_units_origins)

        batch_normalization_visualizer.layout_data.set_input_units_origins(previous_layer_units_origins)

        self.layers[index] = batch_normalization_visualizer

    @dispatch(BinaryCrossEntropyVisualizer, int)
    def set_visualizer_layer(self, binary_crossentropy_visualizer: BinaryCrossEntropyVisualizer, index: int):
        previous_layer_layout_data: LayerVisualizerLayoutData = self.layers[index-1].layout_data
        print(self.layers)

        dense_units_origins = binary_crossentropy_visualizer.layout_data.get_units_origins()
        previous_layer_units_origins = previous_layer_layout_data.get_units_origins()

        previous_layer_layout_data.set_output_units_origins(dense_units_origins)
        binary_crossentropy_visualizer.layout_data.set_input_units_origins(previous_layer_units_origins)
        x = self.calculate_current_x(index)
        (_, y) = binary_crossentropy_visualizer.layout_data.get_origin()
        binary_crossentropy_visualizer.layout_data.set_origin((x, y))
        self.layers[index] = binary_crossentropy_visualizer


    @dispatch(MeanSquaredErrorVisualizer, int)
    def set_visualizer_layer(self, binary_crossentropy_visualizer: MeanSquaredErrorVisualizer, index: int):
        previous_layer_layout_data: LayerVisualizerLayoutData = self.layers[index-1].layout_data

        dense_units_origins = binary_crossentropy_visualizer.layout_data.get_units_origins()
        previous_layer_units_origins = previous_layer_layout_data.get_units_origins()

        previous_layer_layout_data.set_output_units_origins(dense_units_origins)
        binary_crossentropy_visualizer.layout_data.set_input_units_origins(previous_layer_units_origins)
        x = self.calculate_current_x(index)
        (_, y) = binary_crossentropy_visualizer.layout_data.get_origin()
        binary_crossentropy_visualizer.layout_data.set_origin((x, y))
        self.layers[index] = binary_crossentropy_visualizer

    @dispatch(InputVisualizer, int)
    def set_visualizer_layer(self, input_visualizer: InputVisualizer, index: int):
        if len(self.layers) > 1:
            next_layer_layoyt_data: LayerVisualizerLayoutData = self.layers[1].layout_data
            next_layer_layoyt_data.set_input_units_origins(input_visualizer.layout_data.get_units_origins())
            print('updating input')

        self.layers[0] = input_visualizer

    def translate_x(self, origin: tuple[float, float], dx: float):
        (x, y) = origin

        return (x + dx, y)
