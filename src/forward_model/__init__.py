"""Python port of the MATLAB SELE simulator, plus the parameter space it is defined over.

The MATLAB originals (``MATLAB SELE Simulation/calc_Sp2.m`` and ``compute_sele_curve.m``)
remain the reference; ``standalones/validate_matlab_port.py`` checks this port against them.
"""

from src.forward_model.parameters import PARAMETER_SPEC, ParameterSpec
from src.forward_model.sele_simulator import simulate_sele

__all__ = ["PARAMETER_SPEC", "ParameterSpec", "simulate_sele"]
