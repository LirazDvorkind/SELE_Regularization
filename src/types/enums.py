from enum import Enum


class RegularizationMethod(Enum):
    NON_UNIFORM_MESH = 'NON_UNIFORM_MESH',
    TOTAL_VARIATION_TEMPLATE = 'TOTAL_VARIATION_TEMPLATE',
    MODEL_SCORE_GRAD = 'MODEL_SCORE_GRAD'

class LFlag(Enum):
    L0 = 'L0',
    L1 = 'L1',
    L2 = 'L2',
