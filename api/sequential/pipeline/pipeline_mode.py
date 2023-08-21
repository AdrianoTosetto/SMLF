from enum import Enum


class PipelineMode(Enum):
    TRAINING = 0,
    TESTING = 1,
    APPLYING = 2,
