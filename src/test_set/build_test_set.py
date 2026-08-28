"""
Build an expanded ground-truth SELE test set from the paper's MATLAB figures.

Until now the method could only be scored against a single ground-truth profile
(``Data/SELE_ground_truth.csv``). ``Data/Tamir_paper_SELE_figs`` holds the paper's own
figures, two of which contain full SELE(z) profiles: a surface-recombination-velocity sweep
and an SRH-lifetime sweep. Extracting them yields 17 additional physically-generated
profiles that span the shape space the score prior is supposed to cover -- surface dip,
peak depth, bulk magnitude -- rather than one operating point.

For every curve we recompute the measurement from the profile itself: the Beer-Lambert
photogeneration matrix G is built from the paper's own optical constants (see
:mod:`src.optical_constants`) and the ELE follows as ``eta = G @ S``. That keeps
each (SELE, ELE) pair exactly self-consistent, which is what makes the set usable as a
reconstruction benchmark.

Two meshes are emitted per curve, because they answer different questions:

* the curve's *own* mesh (the figures use a non-uniform 0.005-350 um grid) -- the honest
  measurement, generated at the resolution the profile was simulated on;
* the solver's linear mesh over ``W`` -- the same profile and measurement discretized the
  way a reconstruction will see them, so a benchmark can separate reconstruction error from
  the discretization gap between the two meshes.

Run:  python -m src.test_set.build_test_set
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from numpy.typing import NDArray

from src import CONFIG
from src.io import load_csv, save_csv
from src.mesh import _compute_front_generation
from src.matlab_fig import FigAxes, read_fig
from src.optical_constants import extinction_at
from src.types.enums import GStorage

_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _ROOT / "Data"
_FIGS_DIR = _DATA_DIR / "Tamir_paper_SELE_figs"
_OUT_DIR = _DATA_DIR / "test_set"

# Linear solver meshes to emit alongside each curve's native mesh. W tracks the score-grad
# solver so the emitted operator is the one a reconstruction actually builds; the resolution
# is the score model's target_length, kept as a constant here so building the set does not
# depend on a DVC-tracked checkpoint being present.
_SOLVER_W_CM = CONFIG.model_score_grad_config.W
_SOLVER_MESH_RESOLUTIONS = (500,)

# The figures plot depth in um and SELE in percent; the rest of the project works in cm and
# in fraction (the scale the score model was trained on, data_max ~ 1.5e-2).
_UM_TO_CM = 1e-4
_PERCENT_TO_FRACTION = 1e-2

_SRV_FIG = _FIGS_DIR / "SELE_w_PR_vs_SRV.fig"
_TAU_FIG = _FIGS_DIR / "SELE_W_2.FIG"
_SRV_KEY_FIG = _FIGS_DIR / "SELE_at_x=0_vs_SRV_blue.fig"

_SRV_PROFILE_POINTS = 36
_TAU_PROFILE_POINTS = 57


@dataclass
class TestCurve:
    curve_id: str
    source: str
    param_name: str
    param_value: float
    param_units: str
    z_cm: NDArray[np.float64]
    sele: NDArray[np.float64]
    # Set when the profile already lives elsewhere in Data/; the index then points at those
    # files instead of the set copying them (the reference curve is 100k points).
    z_file: str = ""
    sele_file: str = ""
    # Whether this curve's own-mesh G is worth committing, declared per curve rather than
    # inferred later from a missing file.
    G_storage: GStorage = GStorage.FILE


def build_test_set() -> List[TestCurve]:
    curves = _extract_srv_curves() + _extract_tau_curves() + [_existing_ground_truth()]
    _write(curves)
    return curves


# --------------------------------------------------------------------------- extraction

def _extract_srv_curves() -> List[TestCurve]:
    """SELE(z) at six surface recombination velocities (with photon recycling)."""
    axes = _first_axes_with_profiles(read_fig(str(_SRV_FIG)), _SRV_PROFILE_POINTS)
    profiles = [line for line in axes.lines if line.x.size == _SRV_PROFILE_POINTS]

    srv_key = _srv_lookup_table()
    curves = []
    for line in profiles:
        surface_sele = line.y[0]
        srv = srv_key[int(np.argmin(np.abs(srv_key[:, 1] - surface_sele))), 0]
        curves.append(TestCurve(
            curve_id=f"srv_{srv:.0e}".replace("+0", "").replace("e0", "e"),
            source=_SRV_FIG.name,
            param_name="SRV",
            param_value=float(srv),
            param_units="cm/s",
            z_cm=line.x * _UM_TO_CM,
            sele=line.y * _PERCENT_TO_FRACTION,
        ))

    identified = {curve.param_value for curve in curves}
    if len(identified) != len(curves):
        raise ValueError(f"SRV labels are not unique: {sorted(identified)}")
    return sorted(curves, key=lambda curve: curve.param_value)


def _srv_lookup_table() -> NDArray[np.float64]:
    """(SRV, SELE(z=0)) pairs from the companion figure -- the key that labels the profiles.

    The profile figure stores no DisplayName, so the only way to recover which curve is
    which SRV is to match its surface value against the marked points of the SELE(z=0)
    versus SRV sweep.
    """
    axes = read_fig(str(_SRV_KEY_FIG))[0]
    markers = [line for line in axes.lines if line.x.size == 1]
    if not markers:
        raise ValueError(f"No marker points found in {_SRV_KEY_FIG.name}")
    return np.array([[line.x[0], line.y[0]] for line in markers], dtype=np.float64)


def _extract_tau_curves() -> List[TestCurve]:
    """SELE(z) at eleven SRH lifetimes.

    The lifetimes live only in the figure's inset (peak position versus tau_SRH), which
    stores its points in the same order as the profiles -- so they are paired by position,
    then cross-checked against each profile's actual peak depth.
    """
    figure = read_fig(str(_TAU_FIG))
    profile_axes = _first_axes_with_profiles(figure, _TAU_PROFILE_POINTS)
    profiles = [line for line in profile_axes.lines if line.x.size == _TAU_PROFILE_POINTS]

    inset = _inset_series(figure, len(profiles))
    tau_values, peak_positions_um = inset.x, inset.y

    curves = []
    for line, tau, expected_peak_um in zip(profiles, tau_values, peak_positions_um):
        actual_peak_um = line.x[int(np.argmax(line.y))]
        if not np.isclose(actual_peak_um, expected_peak_um, rtol=1e-3):
            raise ValueError(
                f"tau={tau} ns: profile peaks at {actual_peak_um:.4f} um but the inset "
                f"reports {expected_peak_um:.4f} um -- profile/inset ordering broke."
            )
        curves.append(TestCurve(
            curve_id=f"tau_{tau:g}ns".replace(".", "p"),
            source=_TAU_FIG.name,
            param_name="tau_SRH",
            param_value=float(tau),
            param_units="ns",
            z_cm=line.x * _UM_TO_CM,
            sele=line.y * _PERCENT_TO_FRACTION,
        ))
    return curves


def _first_axes_with_profiles(figure: List[FigAxes], profile_points: int) -> FigAxes:
    """Pick the axes holding the depth profiles.

    Both SELE figures draw the same profiles twice (a zoomed axes plus a full-thickness
    one), so taking the first match avoids emitting every curve two times.
    """
    for axes in figure:
        if any(line.x.size == profile_points for line in axes.lines):
            return axes
    raise ValueError(f"No axes with {profile_points}-point profiles found")


def _inset_series(figure: List[FigAxes], expected_points: int):
    for axes in figure:
        for line in axes.lines:
            if line.x.size == expected_points and np.all(np.isfinite(line.x)):
                if not np.array_equal(line.x, np.sort(line.x)):
                    continue
                return line
    raise ValueError(f"No inset series with {expected_points} finite ascending points found")


def _existing_ground_truth() -> TestCurve:
    """The profile the project has always used, carried into the set unchanged."""
    return TestCurve(
        curve_id="paper_gt",
        source="Data/SELE_ground_truth.csv",
        param_name="reference",
        param_value=float("nan"),
        param_units="",
        z_cm=load_csv(str(_DATA_DIR / "z_mesh.csv")).ravel(),
        sele=load_csv(str(_DATA_DIR / "SELE_ground_truth.csv")).ravel(),
        z_file="z_mesh.csv",
        sele_file="SELE_ground_truth.csv",
        # 100k depths make this G 28x100000 -- 70 MB as text, 22 MB packed. It rebuilds from
        # the optical constants in ~0.5 s, so none of that belongs in the repo.
        G_storage=GStorage.COMPUTED,
    )


# ------------------------------------------------------------------- forward simulation

def edges_from_samples(z_cm: NDArray[np.float64]) -> NDArray[np.float64]:
    """Turn sample positions into mesh edges, so G's columns line up with the SELE values.

    G is defined per *element*, but the figures give SELE sampled *at* depths. Placing the
    edges midway between consecutive samples keeps element i centred on sample i; the front
    edge is pinned to the surface and the back edge mirrors the last half-step.
    """
    z_cm = np.asarray(z_cm, dtype=np.float64)
    if z_cm.ndim != 1 or z_cm.size < 2:
        raise ValueError("z_cm must be 1D with at least two samples.")
    if not np.all(np.diff(z_cm) > 0.0):
        raise ValueError("z_cm must be strictly increasing.")

    midpoints = 0.5 * (z_cm[:-1] + z_cm[1:])
    back_edge = z_cm[-1] + (z_cm[-1] - midpoints[-1])
    return np.concatenate(([0.0], midpoints, [back_edge]))


def measurement_wavelengths() -> NDArray[np.float64]:
    return load_csv(str(_DATA_DIR / "wavelength_nm.csv")).ravel()


def photogeneration_matrix_on_edges(z_edges_cm: NDArray[np.float64]) -> NDArray[np.float64]:
    """Beer-Lambert G on a mesh given by its edges, for the measurement wavelengths.

    Attenuation uses the total absorption coefficient (Drude term included), but only
    band-to-band absorption frees a carrier, so each row carries the paper's ``alpha_b/alpha``
    prefactor (SI eq. S4.1). The 4*pi/lambda factors cancel in that ratio, leaving k_b/k.
    """
    wavelengths = measurement_wavelengths()
    k, k_bulk = extinction_at(wavelengths)

    generation = _compute_front_generation(
        k=k,
        wavelength_nm=wavelengths,
        z_cm=z_edges_cm,
        volumetric=False,
    )
    return (k_bulk / k)[:, np.newaxis] * generation


def photogeneration_matrix(z_cm: NDArray[np.float64]) -> NDArray[np.float64]:
    """G on a curve's own depth mesh, whose columns line up with its SELE samples."""
    return photogeneration_matrix_on_edges(edges_from_samples(z_cm))


def solver_mesh_edges(resolution: int) -> NDArray[np.float64]:
    """Edges of the linear solver mesh: ``resolution`` elements spanning W."""
    return np.linspace(0.0, _SOLVER_W_CM, resolution + 1)


def resample_to_edges(curve: TestCurve, z_edges_cm: NDArray[np.float64]) -> NDArray[np.float64]:
    """The curve's SELE at the element centres of another mesh -- interpolated by depth.

    The first centre sits half an element deep, so the surface sample of a figure profile
    (5 nm) is below the finest solver element and cannot be represented; the value there is
    an interpolation between the surface and the next sample, not the surface value itself.
    """
    z_centres = 0.5 * (z_edges_cm[:-1] + z_edges_cm[1:])
    return np.interp(z_centres, curve.z_cm, curve.sele)


def simulate_ele(curve: TestCurve) -> NDArray[np.float64]:
    return photogeneration_matrix(curve.z_cm) @ curve.sele


# ------------------------------------------------------------------------------ writing

def _write(curves: List[TestCurve]) -> None:
    index_rows = ["curve_id,source,param_name,param_value,param_units,n_points,"
                  "sele_surface,sele_peak,peak_position_um,z_file,sele_file,G_storage,G_file"]
    solver_operators = _write_solver_meshes()

    for curve in curves:
        # Paths in the index are relative to Data/, so a curve can point at a profile that
        # already lives outside the test set.
        relative_directory = Path("test_set") / "curves" / curve.curve_id
        directory = _DATA_DIR / relative_directory
        save_csv(str(directory / "ele.csv"), simulate_ele(curve))

        z_file = curve.z_file or str(relative_directory / "z_cm.csv")
        sele_file = curve.sele_file or str(relative_directory / "sele.csv")
        if not curve.z_file:
            save_csv(str(_DATA_DIR / z_file), curve.z_cm)
            save_csv(str(_DATA_DIR / sele_file), curve.sele)

        G_file = _save_native_G(relative_directory, curve)

        for resolution, G_solver in solver_operators.items():
            sele_solver = resample_to_edges(curve, solver_mesh_edges(resolution))
            save_csv(str(directory / f"sele_{resolution}.csv"), sele_solver)
            save_csv(str(directory / f"ele_{resolution}.csv"), G_solver @ sele_solver)

        peak = int(np.argmax(curve.sele))
        index_rows.append(
            f"{curve.curve_id},{curve.source},{curve.param_name},{curve.param_value:g},"
            f"{curve.param_units},{curve.z_cm.size},{curve.sele[0]:.6e},"
            f"{curve.sele[peak]:.6e},{curve.z_cm[peak] / _UM_TO_CM:.4f},"
            f"{z_file.replace(chr(92), '/')},{sele_file.replace(chr(92), '/')},"
            f"{curve.G_storage.value},{G_file.replace(chr(92), '/')}"
        )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUT_DIR / "index.csv").write_text("\n".join(index_rows) + "\n", encoding="utf-8")


def _save_native_G(relative_directory: Path, curve: TestCurve) -> str:
    """Honour the curve's ``G_storage`` choice; return its path relative to ``Data/``.

    A ``COMPUTED`` curve writes nothing and gets an empty path -- the loader rebuilds its G
    through this same module, so the stored and computed paths cannot drift apart.
    """
    if curve.G_storage is GStorage.COMPUTED:
        return ""
    if curve.G_storage is not GStorage.FILE:
        raise ValueError(f"{curve.curve_id}: unhandled G_storage {curve.G_storage!r}")

    G_file = relative_directory / "G.csv"
    save_csv(str(_DATA_DIR / G_file), photogeneration_matrix(curve.z_cm))
    return str(G_file)


def _write_solver_meshes() -> dict:
    """Emit the solver mesh and its G once, and return the operators keyed by resolution.

    On a fixed solver mesh G depends only on the optics, so it is identical for every curve
    -- stored once rather than eighteen times, the same indirection the reference profile
    uses. What does vary per curve is the resampled SELE and the ELE it produces, and those
    are written into each curve's own directory.
    """
    operators = {}
    for resolution in _SOLVER_MESH_RESOLUTIONS:
        z_edges = solver_mesh_edges(resolution)
        operators[resolution] = photogeneration_matrix_on_edges(z_edges)
        save_csv(str(_OUT_DIR / "solver_mesh" / f"z_{resolution}.csv"), z_edges)
        save_csv(str(_OUT_DIR / "solver_mesh" / f"G_{resolution}.csv"), operators[resolution])
    return operators


if __name__ == "__main__":
    built = build_test_set()
    print(f"Wrote {len(built)} ground-truth curves to {_OUT_DIR}")
    for item in built:
        print(f"  {item.curve_id:<12} {item.param_name}={item.param_value:g} "
              f"{item.param_units:<5} n={item.z_cm.size}")
