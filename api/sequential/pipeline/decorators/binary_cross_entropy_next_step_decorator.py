import numpy as np

from api.common_layer import CommonLayer
from api.losses.binary_cross_entropy import BinaryCrossEntropy
from api.losses.common_loss import CommonLoss
from api.losses.regression.mean_squared_loss import MeanSquareError
from api.sequential.pipeline.pipeline_context import (
    PipelineContext,
    PipelineState,
)


class BinaryCrossEntropyNextDecorator(BinaryCrossEntropy):
    def __init__(self, decoratee: CommonLayer, index: int) -> None:
        self.decoratee = decoratee
        self.index = index
        units = self.decoratee.ninputs
        self.last = True

    def forward(self, batch_input: np.ndarray, mode: str = 'train') -> None:
        self.decoratee.forward(batch_input)

    def backward(self, backward_derivatives: np.ndarray) -> None:
        self.decoratee.backward(backward_derivatives)

    def next(self, pipeline: PipelineContext):
        input = pipeline.get_current_input()
        targets = pipeline.get_targets()
        state = pipeline.get_state()

        if state == PipelineState.FORWARD:
            self.decoratee.loss = .0
            self.decoratee.forward(input, targets)
            pipeline.add_epoch_loss(self.decoratee.loss)

            pipeline.set_state(PipelineState.BACKWARD)
            pipeline.set_current(self.index)
            pipeline.set_current_input(pipeline.batch_input)
        else:
            targets = pipeline.get_targets()
            self.decoratee.backward(targets)
            self.mean_output_derivatives_wrt_inputs = np.sum(self.decoratee.derivatives_wrt_inputs, axis=0, keepdims=False)
            pipeline.set_current_backward_derivatives(self.decoratee.derivatives_wrt_inputs)
            pipeline.set_current(self.index - 1)

    def index(self) -> int:
        return self.index

    def output(self):
        return self.decoratee.output
