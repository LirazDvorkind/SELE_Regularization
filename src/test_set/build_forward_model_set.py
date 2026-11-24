"""Build the forward-model test set: SELE curves the parametric method's own simulator can make.

    python -m src.test_set.build_forward_model_set

The paper-figure test set (``build_test_set``) holds Tamir Yeshurun's COMSOL profiles, which
carry physics the simulator lacks, so a reconstruction error there mixes method error with
model mismatch. Every curve here comes out of ``src.forward_model`` itself, so a perfect
inverse method would reproduce it exactly: what is left is the method's own error.

Two groups:
  * ``fm_srv_*`` / ``fm_tau_*`` -- one-at-a-time sweeps through the centre of the parameter
    box, at the paper's SRV values and at its lifetimes that fall inside the box, so the
    results read side by side with the paper-figure sweeps.
  * ``fm_rand_*`` -- draws over the whole box from a seed the training set never used. Each
    one's distance to the nearest training point is recorded, against the typical spacing
    between training points, so "held out" is a measured statement rather than a hope.

The measurement is the closed-form ELE, the same route that made the training set, so the
observation carries no discretisation error. Truth SELE is stored on the solver mesh the
reconstruction reports on. The true parameters are stored too, for scoring parameter
recovery after the fact; nothing that runs inference may read them.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from src.forward_model.analytic_ele import simulate_ele
from src.forward_model.ele import mesh_and_operator
from src.forward_model.parameters import PARAMETER_NAMES, PARAMETER_SPEC
from src.forward_model.sele_simulator import simulate_sele
from src.io import save_csv

_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = _ROOT / "Data" / "test_set_forward_model"
_TRAINING_SET = _ROOT / "Data" / "parametric_model" / "datasets" / "parametric_100k_train.npz"

_RESOLUTION = 500
# Distinct from generate_dataset's default of 7, which made the training set.
_RANDOM_SEED = 20261111
_N_RANDOM = 12

_SRV_VALUES_CM_S = (1e3, 1e4, 1e5, 5e5, 1e6, 1e7)
# The paper sweeps 0.1-60 ns; the box starts at 5 ns, so only its upper lifetimes are
# reachable. 150 ns extends the sweep into the half of the box the paper never visits.
_TAU_VALUES_NS = (8.0, 15.0, 20.0, 40.0, 60.0, 150.0)


def _box_centre() -> NDArray[np.float64]:
    return PARAMETER_SPEC.from_normalized(np.zeros(PARAMETER_SPEC.size))


def _tag(value: float, scientific: bool = False) -> str:
    text = f"{value:.0e}" if scientific else f"{value:g}"
    return text.replace("+0", "").replace("+", "").replace(".", "p")


def sweep_parameters() -> Tuple[List[str], NDArray[np.float64]]:
    centre = _box_centre()
    ids, rows = [], []
    for srv in _SRV_VALUES_CM_S:
        row = centre.copy()
        row[PARAMETER_NAMES.index("S")] = srv
        ids.append(f"fm_srv_{_tag(srv, scientific=True)}")
        rows.append(row)
    for tau_ns in _TAU_VALUES_NS:
        row = centre.copy()
        row[PARAMETER_NAMES.index("tau")] = tau_ns * 1e-9
        ids.append(f"fm_tau_{_tag(tau_ns)}ns")
        rows.append(row)
    return ids, np.array(rows)


def random_parameters() -> Tuple[List[str], NDArray[np.float64]]:
    rng = np.random.default_rng(_RANDOM_SEED)
    kept: List[NDArray[np.float64]] = []
    while len(kept) < _N_RANDOM:
        params = PARAMETER_SPEC.sample(1, rng)
        # The same rejection generate_dataset applies: calc_Sp2's 1/(1 - (alpha Ln)^2)
        # can land a draw on a singularity.
        ele = simulate_ele(params)
        if np.isfinite(ele).all() and (ele > 0).all():
            kept.append(params[0])
    return [f"fm_rand_{i:02d}" for i in range(_N_RANDOM)], np.array(kept)


def nearest_training_distance(params: NDArray[np.float64]) -> Tuple[NDArray[np.float64], float]:
    """Distance in normalised space to the nearest training point, and the training set's own
    median nearest-neighbour spacing for scale."""
    training = np.load(_TRAINING_SET)["params_normalized"]
    query = PARAMETER_SPEC.to_normalized(params)
    distances = np.array([np.sqrt(((training - q) ** 2).sum(axis=1)).min() for q in query])

    rng = np.random.default_rng(0)
    probe = training[rng.choice(training.shape[0], 500, replace=False)]
    spacing = []
    for point in probe:
        d = np.sqrt(((training - point) ** 2).sum(axis=1))
        spacing.append(np.partition(d, 1)[1])
    return distances, float(np.median(spacing))


def build() -> None:
    sweep_ids, sweep_params = sweep_parameters()
    random_ids, random_params = random_parameters()
    ids = sweep_ids + random_ids
    params = np.vstack([sweep_params, random_params])
    groups = ["sweep_srv"] * len(_SRV_VALUES_CM_S) + ["sweep_tau"] * len(_TAU_VALUES_NS) \
        + ["random"] * len(random_ids)

    z_cm, _ = mesh_and_operator(_RESOLUTION)
    sele = simulate_sele(params, z_cm)
    ele = simulate_ele(params)
    if not (np.isfinite(sele).all() and np.isfinite(ele).all()):
        raise RuntimeError("a test curve came out non-finite")

    distances, spacing = nearest_training_distance(params)

    (_OUT_DIR / "curves").mkdir(parents=True, exist_ok=True)
    save_csv(str(_OUT_DIR / "z_cm.csv"), z_cm)

    header = ["curve_id", "group", *PARAMETER_NAMES, "nearest_training_distance",
              "sele_surface", "sele_peak", "peak_position_um"]
    lines = [",".join(header)]
    for curve_id, group, p, s, e, d in zip(ids, groups, params, sele, ele, distances):
        directory = _OUT_DIR / "curves" / curve_id
        directory.mkdir(parents=True, exist_ok=True)
        save_csv(str(directory / "sele.csv"), s)
        save_csv(str(directory / "ele.csv"), e)
        peak = int(np.argmax(s))
        lines.append(",".join([curve_id, group, *(f"{v:.6e}" for v in p), f"{d:.4f}",
                               f"{s[0]:.6e}", f"{s[peak]:.6e}", f"{z_cm[peak] / 1e-4:.4f}"]))
    (_OUT_DIR / "index.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {len(ids)} curves to {_OUT_DIR}")
    print(f"training set's median nearest-neighbour spacing: {spacing:.3f} (normalised units)")
    for curve_id, d in zip(ids, distances):
        print(f"  {curve_id:<16} nearest training point at {d:.3f}")


if __name__ == "__main__":
    build()
