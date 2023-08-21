import threading
import time
from abc import abstractmethod
from enum import Enum
from typing import Callable

import qtawesome as qta
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from multipledispatch import dispatch

from api.batch_dense_normalization import BatchDenseNormalization
from api.sequential.pipeline.pipeline import Pipeline
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
from nn_visualizer.layout_data import LayoutData
from nn_visualizer.ui_components.layout_data.default_dense_visualizer_layout_data import DefaultDenseVisualizerLayoutData
from nn_visualizer.ui_components.layout_data.default_dropout_visualizer_layout_data import DefaultDropoutVisualizerLayoutData
from nn_visualizer.ui_components.layout_data.default_input_layout_data import DefaultInputVisualizerLayoutData
from nn_visualizer.ui_components.smlf_components_visualizers.batch_normalization_visualizer.batch_normalization_unit_visualizer import BatchNormalizationUnitVisualizer
from nn_visualizer.ui_components.smlf_components_visualizers.batch_normalization_visualizer.batch_normalization_visualizer import BatchNormalizationVisualizer
from nn_visualizer.ui_components.smlf_components_visualizers.dense_visualizer.visualizer_dense_layer import DenseVisualizer
from nn_visualizer.ui_components.smlf_components_visualizers.dropout_visualizer.dropout_visualizer import DropoutVisualizer
from nn_visualizer.ui_components.smlf_components_visualizers.input_visualizer.input_visualizer import InputVisualizer


class ApplicationState(Enum):
    PLAYING_FULL_TRAINING = 0
    NEXT_STEP_TRAINING = 1
    IDLE_TRAINING = 2
    EDITING = 3

class ApplicationContext():
    def __init__(self, pipeline: Pipeline, update_layer_callback = None, add_layer_callback = None):
        self.pipeline = pipeline
        self.state = ApplicationState.EDITING
        self.play_full_train_thread = None
        self.selected_nodes: list[tuple[int, int]] = []

        def update(editing_input: EditingInput):
            self.update_layer_editing(0, editing_input)
            self.update_layer_callback(0, editing_input)

        self.editing_loss = EditingBinaryCrossEntropy(1)
        self.editing_layers = [EditingInput(1, update_callback=update)]
        self.add_layer_callback = add_layer_callback
        self.update_layer_callback = update_layer_callback

    def play_full_pipeline(self, event: QEvent):
        self.play_full_train_thread = threading.Thread(target=self.thread_test, kwargs=dict(state_fn=self._get_state_fun_))
        self.play_full_train_thread.start()

    def set_state(self, state: ApplicationState):
        self.state = state

    def _get_state_fun_(self):
        return self.state

    def stop_full_pipeline(self, event: QEvent):
        self.state = ApplicationState.IDLE_TRAINING

    def play_training_handler(self, event: QEvent):
        if (self.state == ApplicationState.IDLE_TRAINING):
            self.state = ApplicationState.PLAYING_FULL_TRAINING
            self.play_full_pipeline(event)
        else:
            self.state = ApplicationState.IDLE_TRAINING

    def next_step_click_handler(self, event: QEvent):
        self.pipeline.next()

    def next_step_training_click_handler(self, event: QEvent):
        self.pipeline.next_training()
        index = self.pipeline.current_layer_index

    def thread_test(self, state_fn):
        while state_fn() == ApplicationState.PLAYING_FULL_TRAINING:
            self.pipeline.next_training()
            time.sleep(0.5)

    def set_learning_rate(self, learning_rate: float):
        self.pipeline.set_learning_rate(learning_rate)

    def add_dense_layer(self, ):
        index = len(self.editing_layers)
        previous_layer = self.editing_layers[index - 1]

        def update(editing_dense: EditingDense):
            self.update_layer_editing(index, editing_dense)
            self.update_layer_callback(index, editing_dense)

        editing_dense = EditingDense(units=3, ninputs=previous_layer.units, update_callback=update)
        self.editing_layers.append(editing_dense)
        self.add_layer_callback(editing_dense, index)

    def add_dropout_layer(self):
        index = len(self.editing_layers)
        previous_layer = self.editing_layers[index - 1]

        def update(editing_dropout: EditingDropout):
            self.update_layer_editing(index, editing_dropout)
            self.update_layer_callback(index, editing_dropout)

        editing_dropout = EditingDropout(units=previous_layer.units, update_callback=update)
        self.editing_layers.append(editing_dropout)
        self.add_layer_callback(editing_dropout, index)
    
    def add_sigmoid_layer(self):
        index = len(self.editing_layers)
        previous_layer = self.editing_layers[index-1]

        def update(editing_sigmoid: EditingSigmoid):
            self.update_layer_editing(index, editing_sigmoid)
            self.update_layer_callback(index, editing_sigmoid)

        editing_sigmoid = EditingSigmoid(units=previous_layer.units, update_callback=update)
        self.editing_layers.append(editing_sigmoid)
        self.add_layer_callback(editing_sigmoid, index)

    def add_relu_layer(self):
        index = len(self.editing_layers)
        previous_layer = self.editing_layers[index-1]

        def update(editing_relu: EditingReLU):
            self.update_layer_editing(index, editing_relu)
            self.update_layer_callback(index, editing_relu)

        editing_relu = EditingReLU(units=previous_layer.units, update_callback=update)
        self.editing_layers.append(editing_relu)
        self.editing_loss.set_input_size(editing_relu.get_output_size())
        self.add_layer_callback(editing_relu, index)

    def add_batch_normalization_layer(self):
        index = len(self.editing_layers)
        previous_layer = self.editing_layers[index-1]

        def update(editing_dropout: EditingBatchNormalization):
            self.update_layer_editing(index, editing_dropout)

        editing_batch_normalization = EditingBatchNormalization(units=previous_layer.units, update_callback=update)
        self.editing_layers.append(editing_batch_normalization)
        self.add_layer_callback(editing_batch_normalization, index)

    def add_softmax_layer(self):
        index = len(self.editing_layers)
        previous_layer = self.editing_layers[index-1]

        def update(editing_dropout: EditingSoftmax):
            self.update_layer_editing(index, editing_dropout)

        editing_batch_normalization = EditingSoftmax(units=previous_layer.units, update_callback=update)
        self.editing_layers.append(editing_batch_normalization)
        self.add_layer_callback(editing_batch_normalization, index)

    def update_layer_editing(self, index: int, layer: EditingLayer):
        self.editing_layers[index] = layer
        self.set_layer_dependents(index, layer)
        if index < len(self.editing_layers) - 1:
            for i in range(index+1, len(self.editing_layers)):
                self.set_layer_dependents(i, self.editing_layers[i])

        if index > 0:
            self.set_layer_dependents(index-1, self.editing_layers[index-1])

    @dispatch(int, EditingInput)
    def set_layer_dependents(self, index: int, editing_input: EditingInput):
        next_layer = self.editing_loss if len(self.editing_layers) == 1 else self.editing_layers[1]
        next_layer.set_input_size(editing_input.units)

    @dispatch(int, EditingDense)
    def set_layer_dependents(self, index: int, editing_dense: EditingDense):
        next_layer = self.editing_loss if index == len(self.editing_layers) - 1 else self.editing_layers[index+1]

        next_layer.set_input_size(editing_dense.get_output_size())
        # self.set_layer_dependents(index+1, next_layer)

    @dispatch(int, EditingBinaryCrossEntropy)
    def set_layer_dependents(self, index: int, editing_mean_squared_error: EditingBinaryCrossEntropy):
        pass

    @dispatch(int, EditingMeanSquaredError)
    def set_layer_dependents(self, index: int, editing_mean_squared_error: EditingMeanSquaredError):
        pass

    @dispatch(int, EditingDropout)
    def set_layer_dependents(self, index: int, editing_dropout: EditingDropout):
        if index == len(self.editing_layers) - 1:
            return

        next_layer = self.editing_layers[index + 1]
        next_layer.set_input_size(editing_dropout.get_output_size())

    @dispatch(int, EditingReLU)
    def set_layer_dependents(self, index: int, editing_relu: EditingReLU):        
        previous_layer = self.editing_layers[index - 1]
        previous_layer.set_output_size(editing_relu.get_input_size())

        next_layer = self.editing_loss if index == len(self.editing_layers) - 1 else self.editing_layers[index + 1]
        next_layer.set_input_size(editing_relu.get_output_size())

    @dispatch(int, EditingSigmoid)
    def set_layer_dependents(self, index: int, editing_sigmoid: EditingSigmoid):
        previous_layer = self.editing_layers[index - 1]
        previous_layer.set_output_size(editing_sigmoid.get_input_size())
        last_index = len(self.editing_layers) - 1

        next_layer = self.editing_loss if index == last_index else self.editing_layers[index+1]
        next_layer.set_input_size(editing_sigmoid.get_output_size())

    @dispatch(int, EditingSoftmax)
    def set_layer_dependents(self, index: int, editing_softmax: EditingSoftmax):
        previous_layer = self.editing_layers[index - 1]
        previous_layer.set_output_size(editing_softmax.get_input_size())

        next_layer = self.editing_layers[index + 1]
        next_layer.set_input_size(editing_softmax.get_output_size())

    @dispatch(int, EditingBatchNormalization)
    def set_layer_dependents(self, index: int, editing_batch_normalization: EditingBatchNormalization):
        previous_layer = self.editing_layers[index - 1]
        previous_layer.set_output_size(editing_batch_normalization.get_input_size())

        next_layer = self.editing_layers[index + 1]
        next_layer.set_input_size(editing_batch_normalization.get_output_size())

    def trace_dependency_path(self, layer_index: int, node_index: int):
        previous_node_indexes = [node_index]
        dependencies_per_layer: list[tuple[int, set]] = []

        for clayer_index in range(layer_index+1, len(self.editing_layers)):
            current_layer = self.editing_layers[clayer_index]
            temp = set(())

            for previous_layer_node_index in previous_node_indexes:
               temp.update(current_layer.get_nodes_indexes_linked_with_previous_node(previous_layer_node_index))

            dependencies_per_layer.append((clayer_index, temp))
            previous_node_indexes = list(temp)

        return dependencies_per_layer
