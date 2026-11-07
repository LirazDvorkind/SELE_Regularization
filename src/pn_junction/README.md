# P-n junction

Everything about extending SELE extraction from a wafer to a p-n junction.

- `drift_diffusion/` -- the 1-D steady-state device solver (see its README), and `sele.py`,
  which turns a solved device into its SELE and SCE profiles. Both are the response of one
  output (front-escaping PL, or collected current) to extra generation at each depth, taken
  from a single adjoint solve and checked against perturbing one depth at a time.
- `feasibility/` -- a synthetic junction measurement for testing reconstruction methods.
  `truth.py` makes the ground-truth SELE from the solver; `measurement.py` builds the
  Beer-Lambert G, adds noise, and computes how sharp a feature the data can resolve at each
  depth. The reconstruction methods never see the solver.
- `profile_sampler.py` -- the model-free reconstruction: draws many SELE profiles that all
  fit the ELE within the noise, assuming only SELE >= 0 and that neighbouring depths are usually
  close but may jump anywhere. How close and how often it jumps are learned from the data. Its
  docstring lists every assumption.
- `standalones/` -- runnable scripts.

```bash
python -m src.pn_junction.standalones.check_drift_diffusion   # solver + SELE checks
python -m src.pn_junction.standalones.plot_band_diagram       # band diagrams
python -m src.pn_junction.standalones.run_junction_tikhonov   # Tamir's Tikhonov on the junction
python -m src.pn_junction.standalones.run_junction_sampler    # the model-free sampler, with controls and robustness checks
```

## What the synthetic measurement leaves out

The optics are Beer-Lambert in bare GaAs: no window, back-surface field, mirror or
interference (the repo has no transfer-matrix optics). The SELE has no photon recycling, and
the thesis' back-surface field is stood in for by a slow back surface. The noise is assumed
white and relative; it has not been measured (see `plans/measurement-noise-open-questions.md`).
