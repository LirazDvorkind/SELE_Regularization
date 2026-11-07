"""Train the parametric posterior network locally.

    python -m src.regularization.parametric_model.standalones.train_parametric_model

The network is small, so this is usable on CPU. The Colab notebook next to this file runs
the same code on a GPU; keep the two in step.

Reported each epoch:
  * validation NLL -- the objective, and the only thing that says the covariance is learning;
  * 68/95% coverage -- the fraction of held-out truths inside the predicted ellipsoids.
    A network can drive the NLL down by being overconfident on average, and coverage is what
    catches that. Error bars that do not cover are worse than no error bars.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.forward_model.parameters import ParameterSpec
from src.regularization.parametric_model import features as feat
from src.regularization.parametric_model.model_definition import (
    build_parametric_network,
    gaussian_nll,
)

_DATA_DIR = Path(__file__).resolve().parents[4] / "Data" / "parametric_model"


@dataclass
class TrainingConfig:
    data_path: str = str(_DATA_DIR / "datasets" / "parametric_100k_train.npz")
    output_path: str = str(_DATA_DIR / "models" / "parametric_posterior.pt")

    input_mode: str = "log_amplitude_shape"
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    use_layer_norm: bool = True
    use_residual: bool = True

    batch_size: int = 512
    learning_rate: float = 1e-3
    num_epochs: int = 200
    weight_decay: float = 0.0
    validation_fraction: float = 0.1
    seed: int = 42

    input_dim: int = field(default=0)       # filled in from the data
    n_parameters: int = field(default=5)


def load_dataset(config: TrainingConfig):
    payload = np.load(config.data_path, allow_pickle=False)
    ele = payload["ele"]
    targets = payload["params_normalized"]
    spec = ParameterSpec(
        names=tuple(str(n) for n in payload["param_names"]),
        units=tuple(ParameterSpec().units),
        lower=tuple(float(v) for v in payload["param_lower"]),
        upper=tuple(float(v) for v in payload["param_upper"]),
        is_log=tuple(bool(v) for v in payload["param_is_log"]),
    )

    rng = np.random.default_rng(config.seed)
    order = rng.permutation(ele.shape[0])
    n_validation = int(round(config.validation_fraction * ele.shape[0]))
    validation_idx, train_idx = order[:n_validation], order[n_validation:]

    # Standardisation statistics come from the training split alone, so the validation
    # numbers stay an honest estimate of unseen data.
    stats = feat.fit_input_stats(ele[train_idx], config.input_mode)
    inputs = feat.transform(ele, config.input_mode, stats)

    amplitude_spread, shape_spread = feat.describe(ele)
    print(f"  {ele.shape[0]} curves, {ele.shape[1]} wavelengths")
    print(f"  amplitude spread {amplitude_spread:.2f} decades, "
          f"shape spread {shape_spread:.2f} decades")

    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32)
    train = TensorDataset(to_tensor(inputs[train_idx]), to_tensor(targets[train_idx]))
    validation = TensorDataset(to_tensor(inputs[validation_idx]),
                               to_tensor(targets[validation_idx]))
    return train, validation, stats, spec, inputs.shape[1]


@torch.no_grad()
def evaluate(network, loader, device) -> Tuple[float, float, float]:
    """``(NLL, 68% coverage, 95% coverage)`` over a loader."""
    network.eval()
    total_nll, total_n = 0.0, 0
    inside_68, inside_95 = 0, 0
    # Squared Mahalanobis radius containing 68.27% / 95.45% of a 5-D Gaussian.
    from scipy.stats import chi2
    threshold_68 = chi2.ppf(0.6827, df=5)
    threshold_95 = chi2.ppf(0.9545, df=5)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mean, scale_tril = network(x)
        total_nll += gaussian_nll(mean, scale_tril, y).item() * x.shape[0]
        total_n += x.shape[0]

        whitened = torch.linalg.solve_triangular(
            scale_tril, (y - mean).unsqueeze(-1), upper=False).squeeze(-1)
        radius = (whitened ** 2).sum(dim=1)
        inside_68 += int((radius <= threshold_68).sum())
        inside_95 += int((radius <= threshold_95).sum())

    network.train()
    return total_nll / total_n, inside_68 / total_n, inside_95 / total_n


def train(config: TrainingConfig) -> None:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {Path(config.data_path).name}")
    train_set, validation_set, stats, spec, input_dim = load_dataset(config)
    config.input_dim = input_dim

    network = build_parametric_network(asdict(config)).to(device)
    n_params = sum(p.numel() for p in network.parameters())
    print(f"  network: {n_params} parameters, input_dim={input_dim}")

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=4096, shuffle=False)

    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=config.learning_rate * 1e-2)

    started = time.time()
    for epoch in range(1, config.num_epochs + 1):
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            mean, scale_tril = network(x)
            loss = gaussian_nll(mean, scale_tril, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item() * x.shape[0]
            seen += x.shape[0]
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1 or epoch == config.num_epochs:
            nll, cov68, cov95 = evaluate(network, validation_loader, device)
            print(f"  epoch {epoch:4d}/{config.num_epochs}  train {running / seen:8.4f}  "
                  f"val {nll:8.4f}  coverage {cov68 * 100:5.1f}% / {cov95 * 100:5.1f}%  "
                  f"({time.time() - started:.0f}s)", flush=True)

    save(network, config, stats, spec)


def save(network, config: TrainingConfig, stats, spec: ParameterSpec) -> None:
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    state = {k.replace("_orig_mod.", ""): v for k, v in network.state_dict().items()}
    torch.save({
        "model_state_dict": state,
        "config": asdict(config),
        "input_stats": {"mean": np.asarray(stats["mean"]), "std": np.asarray(stats["std"])},
        "param_spec": spec.as_dict(),
    }, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    train(TrainingConfig())
