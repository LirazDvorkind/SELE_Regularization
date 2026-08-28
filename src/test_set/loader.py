"""
Read back the ground-truth test set written by :mod:`src.test_set.build_test_set`.

Each curve carries its own depth mesh, so consumers that work on a fixed solver mesh should
resample with :func:`resample_to_mesh` rather than by index -- the meshes span 350 um while
the solver mesh spans ``W`` (~30 um), and index-based resampling would compress the profile
onto the wrong depths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from numpy.typing import NDArray

from src.io import load_G, load_csv
from src.types.enums import GStorage

_DATA_DIR = Path(__file__).resolve().parents[2] / "Data"
_TEST_SET_DIR = _DATA_DIR / "test_set"


@dataclass
class GroundTruthCurve:
    curve_id: str
    param_name: str
    param_value: float
    param_units: str
    z_cm: NDArray[np.float64]
    sele: NDArray[np.float64]
    ele: NDArray[np.float64]

    @property
    def label(self) -> str:
        if self.param_name == "reference":
            return self.curve_id
        return f"{self.param_name}={self.param_value:g} {self.param_units}".strip()


@dataclass
class SolverMeshCurve:
    """A curve discretized onto a linear solver mesh, with the operator that produced it.

    ``ele`` here is ``G @ sele`` on this mesh, so the triple is exactly self-consistent --
    unlike :attr:`GroundTruthCurve.ele`, which was generated on the curve's native mesh and
    therefore differs by the discretization gap between the two.
    """

    curve_id: str
    resolution: int
    z_edges_cm: NDArray[np.float64]
    G: NDArray[np.float64]
    sele: NDArray[np.float64]
    ele: NDArray[np.float64]

    @property
    def z_centres_cm(self) -> NDArray[np.float64]:
        return 0.5 * (self.z_edges_cm[:-1] + self.z_edges_cm[1:])


def _index_rows() -> List[dict]:
    """The index as dicts keyed by column name, so adding a column cannot shift the rest."""
    index_path = _TEST_SET_DIR / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} not found -- run `python -m src.test_set.build_test_set` first."
        )

    lines = index_path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split(",")
    return [dict(zip(header, line.split(","))) for line in lines[1:]]


def _index_row(curve_id: str) -> dict:
    for row in _index_rows():
        if row["curve_id"] == curve_id:
            return row
    raise KeyError(f"{curve_id!r} is not in the test set index.")


def _build_curve(row: dict) -> GroundTruthCurve:
    curve_id = row["curve_id"]
    return GroundTruthCurve(
        curve_id=curve_id,
        param_name=row["param_name"],
        param_value=float(row["param_value"]),
        param_units=row["param_units"],
        # z/sele paths are relative to Data/ -- a curve may reference a profile that
        # already lives outside the test set rather than a copy of it.
        z_cm=load_csv(str(_DATA_DIR / row["z_file"])).ravel(),
        sele=load_csv(str(_DATA_DIR / row["sele_file"])).ravel(),
        ele=load_csv(str(_TEST_SET_DIR / "curves" / curve_id / "ele.csv")).ravel(),
    )


def load_test_set(include_reference: bool = True) -> List[GroundTruthCurve]:
    """Load every curve listed in ``Data/test_set/index.csv``, in index order.

    G is left out deliberately: the reference curve's is 28x100000, too heavy to carry on
    every load. Fetch it per curve with :func:`load_native_G`.
    """
    return [
        _build_curve(row) for row in _index_rows()
        if include_reference or row["param_name"] != "reference"
    ]


def load_curve(curve_id: str) -> GroundTruthCurve:
    """The single named curve, e.g. for driving the pipeline against one test-set profile."""
    return _build_curve(_index_row(curve_id))


def load_native_G(curve_id: str) -> NDArray[np.float64]:
    """The curve's photogeneration matrix on its own depth mesh, as ``eta = G @ sele``.

    Each curve declares in the index whether its G was committed or is to be rebuilt, and
    this dispatches on that declaration. Rebuilding goes through the builder's own
    ``photogeneration_matrix``, so the two paths cannot drift apart; it costs ~0.5 s even for
    the 28x100000 reference.
    """
    row = _index_row(curve_id)
    storage = GStorage(row["G_storage"])

    if storage is GStorage.FILE:
        return np.atleast_2d(load_G(str(_DATA_DIR / row["G_file"])))
    if storage is GStorage.COMPUTED:
        # Imported here rather than at module scope: building G reaches src.mesh, which pulls
        # in matplotlib, and merely reading the test set should not.
        from src.test_set.build_test_set import photogeneration_matrix
        return photogeneration_matrix(load_csv(str(_DATA_DIR / row["z_file"])).ravel())
    raise ValueError(f"{curve_id}: unhandled G_storage {storage!r}")


def load_on_solver_mesh(curve_id: str, resolution: int = 500) -> SolverMeshCurve:
    """Load a curve as the solver sees it: linear mesh, its G, and the matching SELE/ELE.

    G lives once under ``solver_mesh/`` rather than per curve -- on a fixed mesh it depends
    only on the optics, so every curve shares it.
    """
    mesh_dir = _TEST_SET_DIR / "solver_mesh"
    curve_dir = _TEST_SET_DIR / "curves" / curve_id
    for path in (mesh_dir / f"G_{resolution}.csv", curve_dir / f"sele_{resolution}.csv"):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- rebuild with `python -m src.test_set.build_test_set`, "
                f"adding {resolution} to _SOLVER_MESH_RESOLUTIONS if it is not there."
            )

    return SolverMeshCurve(
        curve_id=curve_id,
        resolution=resolution,
        z_edges_cm=load_csv(str(mesh_dir / f"z_{resolution}.csv")).ravel(),
        G=np.atleast_2d(load_csv(str(mesh_dir / f"G_{resolution}.csv"))),
        sele=load_csv(str(curve_dir / f"sele_{resolution}.csv")).ravel(),
        ele=load_csv(str(curve_dir / f"ele_{resolution}.csv")).ravel(),
    )


def resample_to_mesh(curve: GroundTruthCurve, z_edges: NDArray[np.float64]) -> NDArray[np.float64]:
    """Interpolate a curve onto the element centres of a solver mesh given by its edges.

    For the standard solver meshes prefer :func:`load_on_solver_mesh`, which returns the
    same profile alongside the G and ELE that go with it. Note that the first element centre
    sits deeper than a figure profile's surface sample, so the surface value is not
    representable on a coarse mesh.
    """
    z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])
    return np.interp(z_centres, curve.z_cm, curve.sele)
