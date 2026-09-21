"""Turning an ELE measurement into network input.

Shared by training and inference on purpose: a mismatch between the two is silent and would
look like a badly trained network rather than a bug.

Why the amplitude is split off. Across the training set the overall ELE level spans several
decades, while within a single curve the variation across wavelength is only a factor of a
few. Those are two very different quantities. Standardising each wavelength channel
independently would rescale the near-flat channels enormously and hand the network mostly
amplified rounding. Taking a log turns the amplitude spread into an additive offset, and
subtracting it leaves a shape the network can read on its own terms.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

INPUT_MODES = ("log_amplitude_shape", "log_standardized")


def _log_ele(ele: NDArray[np.float64]) -> NDArray[np.float64]:
    ele = np.atleast_2d(np.asarray(ele, dtype=np.float64))
    if np.any(ele <= 0):
        raise ValueError("ELE must be strictly positive to take a logarithm")
    return np.log10(ele)


def raw_features(ele: NDArray[np.float64], input_mode: str) -> NDArray[np.float64]:
    """Pre-standardisation features, ``(n, input_dim)``."""
    log_ele = _log_ele(ele)
    if input_mode == "log_amplitude_shape":
        amplitude = log_ele.mean(axis=1, keepdims=True)
        return np.concatenate([amplitude, log_ele - amplitude], axis=1)
    if input_mode == "log_standardized":
        return log_ele
    raise ValueError(f"unknown input_mode {input_mode!r}, expected one of {INPUT_MODES}")


def fit_input_stats(ele: NDArray[np.float64], input_mode: str) -> Dict[str, NDArray[np.float64]]:
    features = raw_features(ele, input_mode)
    std = features.std(axis=0)
    # A channel with no variation carries no information; leaving its std at zero would turn
    # the standardisation into a division by zero.
    std = np.where(std < 1e-12, 1.0, std)
    return {"mean": features.mean(axis=0), "std": std}


def transform(
        ele: NDArray[np.float64],
        input_mode: str,
        stats: Dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    features = raw_features(ele, input_mode)
    return (features - np.asarray(stats["mean"])) / np.asarray(stats["std"])


def input_dim(n_wavelengths: int, input_mode: str) -> int:
    return n_wavelengths + 1 if input_mode == "log_amplitude_shape" else n_wavelengths


def describe(ele: NDArray[np.float64]) -> Tuple[float, float]:
    """``(amplitude spread, shape spread)`` in decades -- a quick check on a new dataset."""
    log_ele = _log_ele(ele)
    amplitude = log_ele.mean(axis=1, keepdims=True)
    shape = log_ele - amplitude
    return float(np.ptp(amplitude)), float(np.ptp(shape))
