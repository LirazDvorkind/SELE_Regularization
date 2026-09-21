# Forward model

A NumPy port of the MATLAB SELE simulator, plus the parameter space it is defined over.

The MATLAB originals in `MATLAB SELE Simulation/` remain the reference. This port exists
because the parameter-based method needs the physics in two places MATLAB cannot reach: when
building a training set that stores parameters alongside curves, and at inference time, when
sampled parameters have to become SELE curves inside the Python pipeline.

## The five parameters

`parameters.py` is the single definition of what varies, transcribed from the randomisation
block of `create_training_set.m`. Everything else in the simulator — bandgap narrowing, the
intrinsic concentration, the Auger and radiative lifetimes, the free-carrier blend of the
extinction coefficient — is derived from the doping rather than sampled independently.

Normalised space is `[-1, 1]` per parameter, taking a logarithm first -- every parameter is
now sampled log-uniformly. The bounds are exact rather than estimated, so the transform
carries no dataset statistics with it.

One bound departs from the MATLAB original: `D` reaches down to 5 rather than 50 cm^2/s.
`D` and `tau` reach the curve shape only through `Ln = sqrt(D tau_eff)`, so they are
interchangeable there, but `tau` also sets brightness on its own -- which makes a floor on `D`
a ceiling on brightness at fixed shape. At 50 that ceiling sat an order of magnitude below
what the test set's lifetime sweep needs. `standalones/probe_parameter_bounds.py` re-derives
this one bound at a time, and the other probe scripts beside it are the evidence trail.

## Two routes to ELE, and why

`sele_simulator.py` evaluates SELE on a depth mesh. `ele.py` then applies G. That is the
obvious route, and it is the expensive one: the intermediate array is
`curves x emission wavelengths x depths`, and the depth axis has to be fine for the
quadrature to converge. Generating 100000 curves that way took around 27 minutes.

`analytic_ele.py` avoids it. SELE is a sum of exponentials in depth and G's entries are
exponentials in depth, so the sum over mesh elements is a geometric series with a closed
form. Every exponential factorises across the two indices, so nothing transcendental is ever
evaluated on the combined tensor. The same 100000 curves take about 7 seconds, and because
the cost no longer depends on the mesh, the default is a million elements — within 1e-7 of
the continuum limit, where the 500-element solver mesh is off by as much as 17%.

This is exact, not an approximation, and `standalones/check_analytic_ele.py` asserts the two
routes agree. They match to 1e-13 on moderate meshes. Past about 20000 elements the
*explicit* route is the one that drifts, because it adds up that many terms in floating
point; the closed form does a fixed amount of arithmetic whatever the mesh.

Use `analytic_ele` for anything that only needs the measurement. Use `sele_simulator` for
curves you actually want to look at, on a mesh of a few hundred points.

Both are threaded over curve blocks via `_parallel.py`. NumPy ufuncs release the GIL, so
threads scale without pickling arrays between processes, and results are bit-identical
regardless of worker count.

## Where the optics come from

`src/optical_constants.py`, as everywhere else in this project. The port reads the emission
wavelength grid from `Incident_wavelength_dependent_PL.mat`, but never the optical constants
from `optical_constnats_w_wo_Drude.mat` — those two sources are verified bit-identical by the
validation script, and keeping one of them authoritative is what stops the simulator and the
solver drifting apart.

The attenuation-versus-generation distinction matters here as much as in `mesh.py`: `alpha`
carries the Drude term and sets how fast light dies away, `alpha_b` does not and sets what
actually emits. The `alpha_scale` parameter multiplies only `alpha`.

G itself is not reimplemented. `ele.py` takes it from `src/test_set/build_test_set.py`, the
same operator `src/mesh.py` builds, so a benchmark measures reconstruction error rather than
the difference between two G's. `analytic_ele.py` uses G's closed form directly, which is
exactly why it is checked against the real matrix rather than trusted.

## Checks

```bash
cd "MATLAB SELE Simulation" && matlab -batch export_validation_curves
python -m src.forward_model.standalones.validate_matlab_port   # 4.4e-13 vs MATLAB
python -m src.forward_model.standalones.check_analytic_ele     # 1.3e-12 between routes
python -m src.forward_model.standalones.check_dataset          # distribution vs MATLAB curves
```

Re-run the first two after any change to the simulator or to `optical_constants.py`. A
physics port that is only nearly right produces curves that look entirely plausible.

## Probes

`standalones/probe_*.py` are diagnostics rather than checks: they take no position on whether
the code is right, they characterise what the forward model and its parameter box can and
cannot do. Each answers one question and prints its own evidence.

| script | question |
|---|---|
| `probe_ridge.py` | among parameters that fit a measurement equally, how much do the SELE profiles differ, and at what depth? |
| `probe_gaussian_fit.py` | can a single Gaussian stand in for that set? |
| `probe_mixture_size.py` | how many Gaussians would it take? |
| `probe_amplitude.py` | when a curve will not fit, is it the shape or the brightness? |
| `probe_parameter_bounds.py` | which bound is responsible? |

They exist because each backs a decision that is live in the code -- the `D` floor, the choice
not to treat the network's Gaussian as the answer -- and a number nobody can regenerate is an
assertion rather than evidence. `plans/parametric-model-findings.md` records what they said
and what followed from it.
