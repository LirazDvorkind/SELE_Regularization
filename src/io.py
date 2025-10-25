"""Data loading and saving utilities."""
from __future__ import annotations
from dataclasses import is_dataclass, fields
from datetime import datetime
from typing import Any
import numpy as np, os, torch
from src.__init__ import CONFIG
from src.types.enums import RegularizationMethod


def _load_csv_vector(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=',').squeeze()


def load_eta(path: str) -> np.ndarray: return _load_csv_vector(path)
def load_G(path: str) -> np.ndarray: return np.load(path) if path.endswith('.npy') else np.loadtxt(path, delimiter=',')
def load_csv(path: str) -> np.ndarray: return _load_csv_vector(path)


def save_csv(path: str, array: np.ndarray, header: str | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, array, delimiter=',', header='' if header is None else header, comments='')


def load_L_network(path: str):
    net = torch.load(path, weights_only=False)
    net.eval()
    return net


def load_score_model_S() -> np.ndarray:
    return _load_csv_vector(CONFIG.data_paths.score_model_curve)


def generate_run_report(path: str,
                        kappa_knee: float | None = None,
                        kappa1_knee: float | None = None,
                        kappa2_knee: float | None = None) -> str:
    """Generate run report compatible with both 1-D and 2-D modes."""
    def _fmt(v: Any) -> str:
        if hasattr(v, "name") and hasattr(v, "value"): return v.name
        return f"{v}"

    def _flatten(key: str, obj: Any):
        if is_dataclass(obj):
            for f in fields(obj): yield from _flatten(f"{key}{f.name}.", getattr(obj, f.name))
        elif isinstance(obj, dict):
            for k, v in obj.items(): yield from _flatten(f"{key}{k}.", v)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj): yield from _flatten(f"{key}{i}.", v)
        else: yield key[:-1], obj

    method = CONFIG.regularization_method
    exclude = ["model_scoring_params"] if method is RegularizationMethod.NON_UNIFORM_MESH else ["non_uniform_mesh_params"]
    exclude.append("data_paths")

    lines = [f"Run report - {datetime.now().isoformat(timespec='seconds')}",
             f"{'regularization_method':50}: {method.name}"]
    if kappa_knee is not None: lines.append(f"{'κ_knee':50}: {kappa_knee:.3e}")
    if kappa1_knee is not None: lines.append(f"{'κ1_knee':50}: {kappa1_knee:.3e}")
    if kappa2_knee is not None: lines.append(f"{'κ2_knee':50}: {kappa2_knee:.3e}")

    for f in fields(CONFIG):
        if f.name in exclude or f.name == "regularization_method": continue
        for k, v in _flatten(f"{f.name}.", getattr(CONFIG, f.name)):
            lines.append(f"{k:50}: {_fmt(v)}")

    report = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.isdir(path) or path.endswith(os.sep):
        path = os.path.join(path, "run_report.txt")
    with open(path, "w", encoding="utf-8") as fh: fh.write(report)
    print(report, end="")
    return report
