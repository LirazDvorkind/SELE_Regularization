# Status update (2026-11-02): Soreq archive rerun, parametric box check

Follows `2026-10-27-pr-reflection-findings.md` and `2026-10-27-soreq-pin-tikhonov.md`.

## 1. Parametric fits are not held back by the parameter box

The open question from the PR/reflection findings: do the parametric best fits for `tau_60ns` and
`srv_1e7` sit at the box limits? Rerun of `run_inference` (output to a scratch folder, committed
figures untouched; best-draw misfits 0.189% and 0.184% match the committed figures).

| parameter | box | tau_60ns | srv_1e7 |
|---|---|---|---|
| p0 [cm^-3] | 1e16 – 1e19 | 9.4e17 | 5.3e17 |
| D [cm^2/s] | 5 – 200 | 10.9 | 6.7 |
| S [cm/s] | 200 – 1e7 | 5.7e5 | 3.6e5 |
| tau [s] | 5e-9 – 2.5e-7 | 2.3e-8 | 6.9e-8 |
| alpha_scale | 0.1 – 10 | 0.37 | 1.73 |

- No parameter is at a bound (closest: D = 6.7 against 5 for `srv_1e7`).
- The fits reproduce the ELE to 0.2% with parameters that are not the truth's (tau 23 ns vs
  60 ns; S 3.6e5 vs 1e7; D 7–11 vs 38.7 from the Einstein relation).
- From the committed figures: for `srv_1e7` the envelope of draws within 1% reaches the truth
  beyond ~12 um; for `tau_60ns` the whole envelope stays below it beyond ~2 um.
- Hypothesis, not confirmed: the deep shortfall comes from the recycling-free simulator's
  shape being compensated by off-truth parameters, not from the box. PR alone still does not
  account for the whole deficit (see the 10-27 findings).

## 2. Thesis: Tikhonov is never used on a junction

Confirmed by searching the thesis text. Tikhonov (standard, wide-range or projected) is used for the GaAs and InP
wafers and the MOS capacitor (wide-range, L0, 0–1.5 um). The PN-junction section (§3.4) is
forward simulation only (COMSOL, TMM, perturb-and-measure). Inverting a measured junction ELE
is not done in the thesis or the paper.

## 3. Soreq archive: rerun

**Contents.** Older MATLAB scripts that invert one measured PLQY (= ELE) of a PIN sample,
the measurement, GaAs n,k with Drude, and three `.fig` files of results. `G_matrices.mat`
(not listed in the 10-27 note) is a Beer–Lambert G on 0–2 um. It is identical to Beer–Lambert
from the archive's own n,k, so it does not describe the sample's layers.

**Measurement.** One front-side PLQY spectrum: 64 points, 400–695 nm, 1.4e-5 to 1.3e-4.
No error bars, bias, temperature or sample description.

**Setup.** The sample is a PIN device (by name), but the model is a uniform, semi-infinite
GaAs slab solved only over its top 0.5–1.5 um. G = (1-R) x Beer–Lambert with a single alpha
(no `alpha_b/alpha`). Regulariser L0 (penalises magnitude, not curvature). Hansen's
`l_corner`, then the mean over a ±0.25-decade kappa window. Multiplying G by (1-R) implies the
PLQY is per incident photon (a reading of the code, not documented).

**Same method family as `NON_UNIFORM_MESH`.** Differences: L0 vs L2, absorption vs
generation G, short linear mesh vs non-uniform, max-curvature corner vs distance-to-origin.

**Rerun.** The scripts need Hansen's Regularization Tools (v4.1, from MathWorks File Exchange) and
three plotting helpers, replaced by minimal stand-ins. Run in MATLAB R2025b, ~28 s each, from a
scratch folder (nothing committed).

| script | band, domain, cells | PLQY refit misfit | SELE | matches archive `.fig` |
|---|---|---|---|---|
| `_good` | 390–650 nm, 1.5 um, 50 | 1.9% (worst point 17%) | plateau ~2.5e-4 at 0.2–0.45 um, decays to ~0.2e-4 at 1.5 um; spike at 0.05 um, one negative cell | yes, to 1e-13 |
| main | 390–500 nm, 0.5 um, 100 | 3.1% (worst point 22%) | jagged, peak 3.7e-4 at 0.19 um, surface −3.1e-4 | no, differs by up to half the peak |

- The main script's archived 0.5 um curves were made with settings other than the script's
  as committed (unknown which).
- No ground truth exists for this sample, so the result can only be judged by its refit and
  its stability across meshes. Neither of the two runs is mesh-stable (the 10-27 note shows the
  peak moving between 0.04 and 0.45 um with mesh size).

**Use for us.** A real measured ELE to test our methods on, with no ground truth. It does not
provide a junction-aware model.

## Open

1. What doping and diffusivity the paper's sweep figures used. Needed to judge the parametric
   fits and to repeat the PR estimate at the right operating point.
2. Run this repo's Tikhonov (L2, generation G) on the Soreq PLQY and compare with the
   archive's `_good` result and its mesh sensitivity.
3. Ask Gideon or Tamir for the PIN sample's layer stack and how the PLQY was normalised.
