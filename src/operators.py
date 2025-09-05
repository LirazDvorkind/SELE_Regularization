"""Derivative / regularization operators."""
import numpy as np

from src.types.enums import LFlag


def build_L(flag: LFlag, N: int) -> np.ndarray:
    """Return the regularization matrix L.

    Parameters
    ----------
    flag
        'L0', 'L1', or 'L2'.
    N
        Length of the unknown vector S.
    """
    if flag == LFlag.L0:
        return np.eye(N)
    if flag == LFlag.L1:
        L = np.zeros((N - 1, N))
        rows = np.arange(N - 1)
        L[rows, rows] = -1.0
        L[rows, rows + 1] = 1.0
        return L
    if flag == LFlag.L2:
        L = np.zeros((N - 2, N))
        rows = np.arange(N - 2)
        L[rows, rows] = 1.0
        L[rows, rows + 1] = -2.0
        L[rows, rows + 2] = 1.0
        return L
    raise ValueError(f"Unknown L flag '{flag}'. Expected {[member.value for member in LFlag]}.")
