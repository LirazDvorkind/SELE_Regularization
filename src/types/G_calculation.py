from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class GInputData:
    """Optical Generation Matrix (G) Input Data"""
    # Extinction coefficient vector k(λ), from MATLAB code
    k: NDArray[np.float64]

    # Wavelengths [nm] involved in alpha (optical constants) calculations, in MATLAB it is called n_k_wavelength
    lambda_for_alpha: NDArray[np.float64]

    # The wavelengths [nm] of eta_ext, in MATLAB it is called wavelength_back,
    wavelengths: NDArray[np.float64]

    # Depth points vector
    z: NDArray[np.float64]
