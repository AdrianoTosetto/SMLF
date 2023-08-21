import logging
import threading
import time
from enum import Enum

import numpy as np

from api.batch_dense_normalization import BatchDenseNormalization
from api.common_layer import CommonLayer
from api.layers.activation.relu import ReLU
from api.layers.activation.sigmoid import Sigmoid
from api.layers.dense import Dense
from api.losses.binary_cross_entropy import BinaryCrossEntropy
from api.losses.regression.mean_squared_loss import MeanSquareError
from api.optimizers.adagrad_optimizer import AdagradOptmizer
from api.sequential.design_patterns.observer.observable import Event, Observable
from api.sequential.pipeline.decorators.batch_normalization_decorator import BatchNormalizationDecorator
from api.sequential.pipeline.decorators.binary_cross_entropy_next_step_decorator import BinaryCrossEntropyNextDecorator
from api.sequential.pipeline.decorators.dense_next_step_decorator import DenseNextStepDecorator as Decorator
from api.sequential.pipeline.decorators.mean_squared_loss_next_decorator import MeanSquaredLossNextDecorator
from api.sequential.pipeline.decorators.relu_decorator import ReLUNextStepDecorator
from api.sequential.pipeline.decorators.sigmoid_decorator import SigmoidNextStepDecorator
from api.sequential.pipeline.pipeline_context import (
    PipelineContext,
    PipelineMode,
    PipelineState,
)


def _build_layers(layers):
    length = len(layers)
    enumerated = enumerate(layers)

    def to_decorator(value):
        index, layer = value

        if isinstance(layer, Dense):
            print(layer)
            return Decorator(layer, index, index == length - 1)

        if isinstance(layer, BatchDenseNormalization):
            return BatchNormalizationDecorator(layer, index, index == length - 1)

        if isinstance(layer, ReLU):
            return ReLUNextStepDecorator(layer, index, last=False)

        if isinstance(layer, Sigmoid):
            return SigmoidNextStepDecorator(layer, index, last=False)

        if isinstance(layer, MeanSquareError):
            return MeanSquaredLossNextDecorator(layer, index)

        if isinstance(layer, BinaryCrossEntropy):
            return BinaryCrossEntropyNextDecorator(layer, index)
    
    return list(map(lambda value: to_decorator(value), enumerated))


class Pipeline(PipelineContext, Observable):
    def __init__(self, layers = [], next_callback = None):
        PipelineState.__init__(self)
        Observable.__init__(self)
        self.api_layers = layers
        self.layers = _build_layers(layers)
        self.state = PipelineState.FORWARD
        self.mode = PipelineMode.TRAINING
        self.current = None
        self.current_input = None
        self.current_backward_derivatives: np.ndarray = None
        self.last_index = len(layers) - 1
        self.next_callback = next_callback
        self.epoch = 0
        self.epochs_losses = []
        self.epochs_validation_losses = []
        self.batch_validation_input = np.ndarray([])
        self.batch_validation_targets = np.ndarray([])
        self.loss_change_listeners = []
        self.learning_rate = 1.0

    def set_pipeline(self, layers, batch, loss_change_listener):
        self.layers = _build_layers(layers)

        self.layers = _build_layers(layers)
        self.state = PipelineState.FORWARD
        self.mode = PipelineMode.TRAINING
        self.current_backward_derivatives: np.ndarray = None
        self.last_index = len(layers) - 1
        self.epoch = 0
        self.epochs_losses = []
        self.loss_change_listeners = [loss_change_listener]
        self.learning_rate = 1.0
        self.set_input(batch)
        self.set_current(0)

    def set_current(self, index: int):
        for i in range(len(self.layers)):
            self.layers[i].is_current = False

        self.current_layer = self.layers[index]
        self.current_layer_index = index
        self.layers[index].is_current = True

    def get_state(self) -> PipelineState:
        return self.state

    def set_state(self, state: PipelineState):
        self.state = state

    def get_current_input(self) -> np.ndarray:
        return self.current_input

    def set_current_input(self, input: np.ndarray):
        self.current_input = input

    def get_targets(self) -> np.ndarray:
        return self.targets

    def set_targets(self, targets: np.ndarray):
        self.targets = targets
        return self

    def get_current_backward_derivatives(self) -> np.ndarray:
        return self.current_backward_derivatives

    def set_current_backward_derivatives(self, derivatives: np.ndarray) -> None:
        self.current_backward_derivatives = derivatives

    def set_input(self, batch_input):
        self.batch_input = batch_input
        self.current_input = batch_input
        self.current_layer = self.layers[0]
        self.state = PipelineState.FORWARD

        return self

    def set_validation_inputs(self, batch_input: np.ndarray):
        self.batch_validation_input = batch_input

    def set_validation_targets(self, targets: np.ndarray):
        self.batch_validation_targets = targets

    def next(self):
        self.current_layer.next(self)

        index = self.current_layer.index + 1
        new_input = self.current_layer.output()

        self.set_current(index)
        self.set_current_input(new_input)

    def next_training(self):
        update_index_callback = self.current_layer_index
        self.next_callback(update_index_callback)

        self.current_layer.next(self)
        self.notify(Event(dict({})))

        if self.current_layer.index == 0 and self.state == PipelineState.FORWARD:
            for layer in self.layers[::-1]:
                if isinstance(layer.decoratee, Dense):
                    opt = AdagradOptmizer(learning_rate=self.learning_rate)
                    opt.update_weights(layer.decoratee)

    def add_epoch_loss(self, loss: float) -> None:
        validation_loss = self.calculate_validation_loss()
        self.epochs_validation_losses.append(validation_loss)
        self.epochs_losses.append(loss)
        for loss_listener in self.loss_change_listeners:
            loss_listener(np.arange(0, self.epoch+1), self.epochs_losses, self.epochs_validation_losses)
        self.epoch += 1

    def calculate_validation_loss(self):
        input = self.batch_validation_input
        for layer in list(map(lambda decorator: decorator.decoratee, self.layers[0:-1])):
            layer.forward(input, mode='predict')
            input = layer.output

        self.layers[-1].decoratee.forward(input, self.batch_validation_targets)

        return self.layers[-1].decoratee.loss

    def get_learning_rate(self) -> float:
        return self.learning_rate

    def set_learning_rate(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

        for layer in self.layers:
            layer.reset_state()
