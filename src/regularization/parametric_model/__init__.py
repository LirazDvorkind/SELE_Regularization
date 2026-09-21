"""Parameter-based SELE reconstruction.

Instead of learning a prior over 500-point SELE curves, infer the five physical parameters
the simulator is defined by, with a full covariance that reports how badly the measurement
determines them. SELE curves and uncertainty bands then come from running the physics
(``src.forward_model``) on samples from that distribution.
"""

from src.regularization.parametric_model.model_definition import (
    ParametricPosteriorNetwork,
    build_parametric_network,
    gaussian_nll,
)

__all__ = ["ParametricPosteriorNetwork", "build_parametric_network", "gaussian_nll"]
