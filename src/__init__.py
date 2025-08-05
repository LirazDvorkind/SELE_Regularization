"""SELE toolbox with non‑uniform mesh support."""

from importlib import import_module as _imp

# Re‑export key public APIs
from .operators import build_L  # noqa: F401
from .pipeline import run_regularization  # noqa: F401
