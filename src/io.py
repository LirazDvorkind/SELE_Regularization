"""Data loading and saving utilities."""
from __future__ import annotations

from dataclasses import is_dataclass, fields

from datetime import datetime
from typing import Any, Iterable
import numpy as np
import os

import torch

from src.__init__ import CONFIG
from src.types.enums import RegularizationMethod


def _load_csv_vector(path: str) -> np.ndarray:
    """Load a 1‑D CSV file and squeeze to shape (N,)."""
    return np.loadtxt(path, delimiter=',').squeeze()


def load_eta(path: str) -> np.ndarray:
    return _load_csv_vector(path)


def load_G(path: str) -> np.ndarray:
    if path.endswith('.npy'):
        return np.load(path)
    return np.loadtxt(path, delimiter=',')


def load_z(path: str) -> np.ndarray:
    return _load_csv_vector(path)


def save_csv(path: str, array: np.ndarray, header: str | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, array, delimiter=',', header='' if header is None else header, comments='')


def load_L_network(path: str) -> np.ndarray:
    score_network = torch.load(path, weights_only=False)
    score_network.eval()
    return score_network


def load_score_model_S() -> np.ndarray:
    # Uncomment these to load the ground truth SELE instead of the model-learned SELE.
    # vect = _load_csv_vector(CONFIG.data_paths.sele_gt)
    # # Interpolate the GT SELE because it is 10000 long
    # return np.interp(np.linspace(0, CONFIG.model_scoring_params.W, CONFIG.model_scoring_params.points_amount),
    #                  np.linspace(0, CONFIG.model_scoring_params.W, len(vect)),
    #                  vect)
    return _load_csv_vector(CONFIG.data_paths.score_model_curve)


def generate_run_report(path: str, kappa_knee: float) -> str:
    def _fmt(v: Any) -> str:
        if hasattr(v, "name") and hasattr(v, "value"):  # Enum
            return v.name
        return f"{v}"

    def _flatten(key: str, obj: Any) -> Iterable[tuple[str, Any]]:
        if is_dataclass(obj):
            for f in fields(obj):
                yield from _flatten(f"{key}{f.name}.", getattr(obj, f.name))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                yield from _flatten(f"{key}{k}.", v)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                yield from _flatten(f"{key}{i}.", v)
        else:
            yield key[:-1], obj

    # Decide which branch to exclude based on method
    method = CONFIG.regularization_method
    exclude_root = ["model_scoring_params"] if method is RegularizationMethod.NON_UNIFORM_MESH else \
        ["non_uniform_mesh_params"] if method is RegularizationMethod.MODEL_SCORING else []
    exclude_root.append("data_paths")

    lines = [f"Run report - {datetime.now().isoformat(timespec='seconds')}",
             f"{'regularization_method':50}: {method.name}",
             f"{'κ_knee':50}: {kappa_knee:.3e}"]
    # walk all top-level fields except the excluded branch (but include the chosen one)
    for f in fields(CONFIG):
        name = f.name
        if name == "regularization_method":
            continue
        if exclude_root and name in exclude_root:
            continue
        for k, v in _flatten(f"{name}.", getattr(CONFIG, name)):
            lines.append(f"{k:50}: {_fmt(v)}")

    report = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.isdir(path) or path.endswith(os.sep):
        path = os.path.join(path, "run_report.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report, end="")
    return report
