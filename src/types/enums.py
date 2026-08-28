from enum import Enum


class RegularizationMethod(Enum):
    NON_UNIFORM_MESH = 'NON_UNIFORM_MESH',
    TOTAL_VARIATION = 'TOTAL_VARIATION_TEMPLATE',
    MODEL_SCORE_GRAD = 'MODEL_SCORE_GRAD'

class LFlag(Enum):
    L0 = 'L0',
    L1 = 'L1',
    L2 = 'L2',


class GStorage(Enum):
    """How a test-set curve's own-mesh G is obtained.

    G is a pure function of the optical constants and the mesh, so committing one is a
    convenience, not a necessity. Each curve states which applies rather than leaving it to
    be inferred from a missing file.
    """
    FILE = 'FILE'          # written to disk; the index carries its path
    COMPUTED = 'COMPUTED'  # rebuilt on load — too large to be worth storing
