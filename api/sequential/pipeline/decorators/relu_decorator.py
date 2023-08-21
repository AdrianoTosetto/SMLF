import numpy as np

from api.common_layer import CommonLayer
from api.layers.activation.relu import ReLU
from api.layers.dense import Dense
from api.optimizers.adagrad_optimizer import AdagradOptmizer
from api.optimizers.stochastic_gradient_descent import StochasticGradientDescent
from api.sequential.pipeline.pipeline_context import (
    PipelineContext,
    PipelineState,
)


class ReLUNextStepDecorator(CommonLayer):
    def __init__(self, decoratee: ReLU, index: int, last: bool = False) -> None:
        self.decoratee = decoratee
        self.index = index
        units = self.decoratee.ninputs
        self.last = last
        CommonLayer.__init__(self, self.decoratee.ninputs, units)

    def forward(self, batch_input: np.ndarray, mode: str = 'train') -> None:
        self.decoratee.forward(batch_input)

    def backward(self, backward_derivatives: np.ndarray) -> None:
        self.decoratee.backward(backward_derivatives)

    def next(self, pipeline: PipelineContext):
        input = pipeline.get_current_input()
        state = pipeline.get_state()

        if state == PipelineState.FORWARD:
            self.decoratee.forward(input)
            pipeline.set_current_input(self.decoratee.output)

            if self.last:
                pipeline.set_state(PipelineState.BACKWARD)
                pipeline.set_current(self.index)
            else:
                pipeline.set_current(self.index + 1)
        else:
            self.decoratee.backward(pipeline.get_current_backward_derivatives())
            pipeline.set_current_backward_derivatives(self.decoratee.derivatives_wrt_inputs)

            self.mean_output_derivatives_wrt_inputs = np.sum(self.decoratee.derivatives_wrt_inputs, axis=0, keepdims=False)

            if self.index == 0:
                pipeline.set_state(PipelineState.FORWARD)
                pipeline.set_current(0)
            else:
                pipeline.set_current(self.index - 1)

    def index(self) -> int:
        return self.index

    def output(self):
        return self.decoratee.output

    def reset_state(self):
        self.decoratee.reset_state()
