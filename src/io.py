"""Data loading and saving utilities."""
from __future__ import annotations

import numpy as np
import os

import torch


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