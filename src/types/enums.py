from enum import Enum


class RegularizationMethod(Enum):
    NON_UNIFORM_MESH = 'NON_UNIFORM_MESH',
    MODEL_SCORING = 'MODEL_SCORING'

class LFlag(Enum):
    L0 = 'L0',
    L1 = 'L1',
    L2 = 'L2',