# Paper figures (MATLAB `.fig`)

Source figures from the paper, kept because they are the only place several of its numbers
exist in machine-readable form. MATLAB is **not** required to use them: `src/test_set/` turns
two of them into `Data/test_set/`, and `src/optical_constants.py` reads the optical constants
out of a third to build **every** G matrix in the project, the solver's included.

Read them with `src/matlab_fig.py` (`read_fig`), a small `scipy.io.loadmat` wrapper.
A `.fig` is a MAT-file holding one `hgS_070000` handle-graphics struct; line data hides in
`properties.XData/YData`, and axis labels are `text` children carrying a `String` field, not
`XLabel`/`Title` properties.

## What each file holds

| File | Contents | Used for |
|------|----------|----------|
| `SELE_w_PR_vs_SRV.fig` | 6 SELE(z) profiles, 36 pts, SRV sweep with photon recycling | `srv_*` test curves |
| `SELE_W_2.FIG` | 11 SELE(z) profiles, 57 pts, **SRH-lifetime** sweep + inset | `tau_*` test curves |
| `SELE_at_x=0_vs_SRV_blue.fig` | SELE(z=0) vs SRV, plus one marker per simulated SRV | labelling the `srv_*` curves |
| `Optical constants GaAs.fig` | n and k, 601 pts over 191–990 nm, with and without Drude | every G matrix, via `src/optical_constants.py` |
| `Optics results corrected PL.fig` | measured + modelled reflectance, transmittance, front/back PL | nothing (yet) |

## Traps worth knowing before re-deriving any of this

- **`SELE_W_2.FIG` is a lifetime sweep, not a thickness sweep.** The name suggests width, but
  its inset axes read `\tau_{SRH} [ns]` vs `Max Position [\mum]`, and a 0.1 µm-thick wafer
  could not peak at 2.18 µm. The lifetimes exist *only* in that inset; profiles are paired to
  it by stored order and then cross-checked against each profile's actual peak depth.
- **Both SELE figures draw their profiles twice** (a zoomed axes plus a full-thickness one).
  Iterating all axes yields every curve twice — `_first_axes_with_profiles` takes the first.
- **The SRV profiles carry no `DisplayName`.** The only way to recover which curve is which
  SRV is to match each profile's surface value against the marker points in
  `SELE_at_x=0_vs_SRV_blue.fig`.
- **The figures plot SELE in percent and depth in µm.** The project works in fraction and cm.
- **`Optics results corrected PL.fig` is MAT v7.3 (HDF5)** and will not open with
  `read_fig`/`loadmat`; it needs `h5py`, which is deliberately *not* a project dependency. It
  was inspected once and holds no SELE and no optical constants, so nothing needs it. Its
  reflectance is not required either — see the optics discussion in
  `Data/test_set/README.md`.
- **`Optical constants GaAs.fig` is bit-identical** to
  `MATLAB SELE Simulation/optical_constnats_w_wo_Drude.mat` (same 601-point grid, zero
  difference on all four series). Either source is fine; no need to reconcile them.
