# Forward-model test set: can the parametric method reconstruct what its own model makes?

A sanity check on the parametric method (`plans/parametric-model.md`), run on 2026-11-11.

## Motivation and what was asked

The parametric method infers the five simulator parameters from an ELE measurement and
reports simulator SELE curves. Until now it was scored only against the ground-truth set in
`Data/test_set/`, which is extracted from Tamir Yeshurun's paper figures. Those truths are
COMSOL simulations with physics our forward model lacks (photon recycling, multi-pass
escape), so every error there mixes two things: the inverse method's own error and model
mismatch.

The user asked for a check that separates them. The request was to generate ground-truth SELE
curves with our own forward model, a reasonable number of them and diverse. Then run the same
reverse parametric method on their ELE and see whether it reconstructs them perfectly.
Explicit conditions:

- **No cheating.** The reconstruction must not know the answer ahead of time: inference gets
  the ELE only, and the curves must not be training examples.
- **Ask, don't assume.** Clarifying questions were asked rather than guessed at (decisions
  below).
- The goal is **not** to make reconstruction perfect at all costs. It is to swap in
  forward-model ground truth and observe what the unchanged method does.

## Decisions taken with the user

| question | decision |
|---|---|
| where the curves live | a **separate** set, `Data/test_set_forward_model/`; the paper-figure set is untouched |
| how curves are chosen | **sweeps + random**: SRV and lifetime one-at-a-time sweeps through the centre of the parameter box, plus 12 draws from a held-out seed |
| which ELE is the measurement | the **closed-form ELE**, the same route that made the training set, so the observation carries no discretisation error (the ideal case) |

No noise was added: deliberately the ideal case.

## What was built

- `src/test_set/build_forward_model_set.py`: builds the 24 curves. SRV sweep at the paper's
  six values, lifetime sweep at 8–150 ns (the paper's lifetimes below 5 ns lie outside the
  box), others at the box centre. The random draws use a seed different from the training
  set's, reject singular draws as `generate_dataset` does, and record their distance to the
  nearest training point. Truth SELE is stored on the 500-element solver mesh.
- `src/test_set/loader.py:load_forward_model_set`: returns `ele` and `true_params`
  separately. The true parameters are for scoring after inference only.
- `score_test_set.py --set forward_model`: the existing per-zone scoring, plus the best draw's
  ELE misfit and a parameter-recovery table (log10 best/true).
- `plot_test_set_results.py`: the consolidated figures below, for both sets.

**Held-out check.** Each test curve's nearest training point is 0.07–0.21 away in normalised
parameter space, against a median spacing of 0.14 between training points. The curves are as
unseen as any point in the box can be.

## Results

Numbers and their interpretation are in `plans/parametric-model.md`, under "Without model
mismatch"; regenerate them with the scorer. In short:

- The ELE is reproduced to 0.01–0.06%.
- SELE is reconstructed essentially exactly over 0–1 µm (≤ 0.2% on every curve) and closely
  over 1–3 µm (< 3%). On the paper-figure set the 1–3 µm error was 5–12%, so most of that was
  model mismatch.
- Past ~3 µm the error grows (up to ~50%) and the **parameters are not recovered** (off by up
  to ~1.5 decades). With 10× more draws the ELE fit improves ~5× but neither the deep error
  nor the parameters converge. The best draw just moves to a different point on the set of
  parameters that fit equally well. This is non-identifiability of the data, not a sampling
  shortfall or a method bias.
- Three curves (`fm_rand_02`, `_06`, `_11`: low doping, long lifetime, diffusion length
  beyond the 30 µm window, SELE peak at 15–25 µm) have truth outside the envelope at depth.
  The proposal never reaches their true lifetime, and only the unmeasured depths would
  reveal it.

## Figures

`python -m src.regularization.parametric_model.standalones.plot_test_set_results` writes them
to `results/parametric_model/test_set_comparison/`. All use the default 4000 draws.

- **`forward_model_curves.png`**: one panel per forward-model curve, linear 0–30 µm depth axis. Truth
  (dashed), best draw (bold), shaded envelope of draws within 1% in ELE; the 0–1 and 1–3 µm
  errors are in each title. Truth and best draw coincide to a few µm. The visible misses are
  the three long-diffusion-length curves, deep only.
- **`paper_curves.png`**: the same layout for the 17 Yeshurun curves, for panel-by-panel
  comparison. On the lifetime sweep the best draw undershoots the truth's peak and the truth
  sits outside the envelope: the missing recycling physics.
- **`paired_sweeps.png`**: each Yeshurun sweep curve (blue) overlaid with its forward-model
  twin at the same SRV or lifetime (pink): truth dashed, best draw solid, band shaded, both
  sets' 0–1 / 1–3 µm errors in the title. The twins share only the swept value, so the curves
  themselves are not expected to coincide; what compares is how closely each set's
  reconstruction follows its own truth. Paper lifetimes 0.1–4 ns have no twin (below the
  parameter box); `tau_150ns` has no paper counterpart.
- **`set_comparison.png`**: left, best-draw SELE error per depth zone for both sets (log
  scale, median bars). The forward-model set is about an order of magnitude lower in every
  zone. Right, 0–3 µm coverage: 100% on 21 of 24 forward-model curves, against 4 of 17 paper
  curves.
- **`parameter_recovery.png`**: forward-model set only. Best-draw parameter against the true
  one per parameter (log–log, identity dotted), with a bar for the range over the draws that
  fit. Most bars span more than a decade and the lifetime bars span the whole box: the
  measurement does not pin the parameters.

## Open

1. **Noise.** Add realistic relative noise (1–3%) to these same ELEs and see how far the
   0–3 µm accuracy degrades. The network was trained noise-free, so this also tests that.
2. **Point the classical methods at this set** (`NON_UNIFORM_MESH`, `TOTAL_VARIATION`). It is
   a clean benchmark with no model mismatch, for comparing methods as well as this one.
3. Whether the reported parameters should be presented at all, given they are not
   identifiable even in the ideal case.
