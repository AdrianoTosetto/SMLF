from enum import Enum


class PipelineState(Enum):
    IDLE = 0 # no input received yet
    FORWARD = 1 # computing input
    BACKWARD = 2
