# Score Model Regularization — Status and Where We Stopped

> **Read this before touching anything in this folder.** It documents the *current state* of the score-model approach, why it accumulated the shape it has, and which decisions are load-bearing vs. accidental. The actual recommended forward direction is in `AI/GENERIC_GUIDELINES.md`, `AI/TV_DIRECTION.md`, and `AI/ML_DIRECTION.md` — and the recommendation is to **not extend this code further**, but to start a fresh solver. This README exists so that you (or future Claude) can pick up where we stopped without re-doing the archeology.

---

## TL;DR

- This folder implements a **Nesterov Accelerated Gradient solver** that combines a data fidelity term (`||GS - B||²`) with a learned **score-based prior** (a small PyTorch MLP that approximates `∇ log p(S)` for SELE curves).
- It works *somewhat* on GaAs synthetic curves from the same analytical generator the model was trained on. It does not generalize, and we have stopped trying to make it generalize via this code path.
- The solver has been heavily reworked. The current head version uses **full-Hessian preconditioning + global scalar score weighting**, which is the third major iteration. The history matters because each fix exposed the next problem.
- Hyperparameters are still hand-tuned per training run. There is no automatic regularization weight selection. There is no uncertainty quantification.
- Several rocks are left unturned (listed below). They are **not** worth turning over in this codepath; they are listed here so you know what was *not* tried, in case you need to revisit one for a specific debugging question.

---

## What the code currently does

The entry point is `solve_gradient_descent()` in `score_model_grad.py`. Pseudocode:

```
Load score network checkpoint (gives score net, d_min, d_max, target_length).
Normalize G by mean(|G|), normalize B the same way.
Precompute H = 2 G_norm.T @ G_norm / norm_scale²,
            H_reg = H + ε I,
            H_inv = inv(H_reg).                # full-Hessian preconditioner
S_norm  = zeros(N)                              # start at center of normalized range
velocity = zeros(N)

for i in 0 .. MAX_STEPS:
    S_lookahead = S_norm + MOMENTUM * velocity
    S_phys      = denormalize(S_lookahead)

    # Data fidelity gradient (in normalized space)
    grad_fidelity_norm = (2 G_norm.T (G_norm @ S_phys - B_norm)) / norm_scale

    # Score model evaluation at fixed small noise level T0
    score = score_network(S_lookahead, T0)

    # Combine
    preconditioned_grad = H_inv @ grad_fidelity_norm
    score_weighted      = REG_WEIGHT * score
    total_update        = preconditioned_grad - score_weighted

    # Cosine-annealed LR + Nesterov momentum
    lr       = cosine_anneal(LR_MAX, LR_MIN, i, MAX_STEPS)
    velocity = MOMENTUM * velocity - lr * total_update
    S_norm   = S_norm + velocity

    # Track MSE vs ground truth, early-stop on stagnation or blowup
```

The early-stop condition has two triggers: `MSE > 1` (the solve has diverged) and `|ΔMSE| < STOP_CHANGE` for `STOP_STEPS` consecutive iterations (convergence stall).

---

## Why the code looks the way it does — the iteration history

This is the part that is hardest to recover from `git log` alone, so it's documented in detail.

### Iteration 0 — vanilla NAG with adaptive scalar weighting
- Original formulation: `total = grad_fidelity_norm - (||grad|| / ||score||) · REG_WEIGHT · score`.
- The `||grad|| / ||score||` factor was added because the score net (trained on normalized data) outputs O(1) vectors while the data gradient could be 10⁻¹⁰ to 10⁵ depending on G's scale.
- **Problem observed:** under-regularized → noisy reconstruction; over-regularized → roughly the right shape but the gradient is dominated by `(||grad||/||score||) · REG_WEIGHT · sign(score)` and the per-iterate score *direction* matters more than its *magnitude*.

### Iteration 1 — per-element adaptive weighting (REMOVED)
- Tried: `score_weighted[j] = (|grad[j]| / |score[j]|) · REG_WEIGHT · score[j]`.
- This was mathematically broken: it collapses to `sign(score[j]) · |grad[j]| · REG_WEIGHT`, which **erases the score's magnitude entirely** and makes the prior contribute only its sign pattern. The result was that for `REG_WEIGHT > 1` every reconstruction converged to nearly the same shape regardless of ground truth (because the sign pattern of the score is roughly the same for any near-prior point), and for `REG_WEIGHT < 1` it was just noisy.
- This is why the user observed "low REG_WEIGHT → noisy, high REG_WEIGHT → same surface every time" — that observation triggered the audit that found the bug.
- **Fix:** removed per-element weighting, replaced with diagonal Hessian preconditioning (Iteration 2).

### Iteration 2 — diagonal Hessian preconditioning
- Tried: `preconditioned_grad = grad_fidelity_norm / diag(G_norm.T @ G_norm)`.
- The intuition was that columns of G have very different norms (Beer-Lambert decay), so preconditioning by the column norms equalizes the gradient's effective step size across depths.
- **Problem observed:** the peak of the reconstructed SELE was underestimated by ~8x and the surface was visibly noisy. Diagnosis: adjacent columns of G **share the boundary terms `exp(-α z_j)`**, so off-diagonal entries of `G^T G` are *not* negligible — there is strong neighbor-to-neighbor coupling that a diagonal preconditioner ignores, causing the solver to bounce on the off-diagonal modes.
- **Fix:** replace diagonal preconditioner with the full Hessian inverse (Iteration 3, current).

### Iteration 3 — full-Hessian preconditioning (CURRENT HEAD)
- Current code: `H_inv = inv(2 G_norm.T G_norm / norm_scale² + ε I)`, then `preconditioned_grad = H_inv @ grad_fidelity_norm`.
- This eliminates the neighbor coupling problem from Iteration 2.
- **New problem observed (and not solved):** preconditioning *correctly* shrinks `||preconditioned_grad||` (because H is ill-conditioned and `H_inv` damps the small-singular-value modes that the data gradient leaned on). After the change, `||preconditioned_grad|| ≈ 5.79` vs `||score|| ≈ 27.9` at init — the score now dominates the data gradient by ~5x at the very first iteration, which is the wrong end of the trade-off.
- The recommended re-tune given this is roughly `REG_WEIGHT ≈ 0.15` (down from 1.5), `LR_MAX ≈ 0.1` (up from 1e-4), `MOMENTUM ≈ 0.5` (down from 0.9). **These have not been validated end-to-end.** This is where we stopped.

The bigger lesson from this history is that each fix exposed the next problem, which is the symptom of a *formulation* problem, not a *constants* problem. That realization is what motivated the broader rethink in the `AI/` direction docs.

---

## Where we stopped — the exact state

- **`score_model_grad.py`** is at Iteration 3 (full-Hessian preconditioning, global scalar score weight). The code in this file is the latest version.
- **`hyperparameter_playground.py`** is the active sandbox. The current settings in it (`REG_WEIGHT=1.5`, `LR_MAX=1e-2`, `LR_MIN=1e-5`, `MOMENTUM=0.1`, `T0=5e-2`) are *not* tuned for Iteration 3 — they are leftover from earlier tries. If you want to re-run this code as it stands, the recommended starting point given the diagnostics is `REG_WEIGHT ≈ 0.15`, `LR_MAX ≈ 0.1`, `MOMENTUM ≈ 0.5`. None of these have been confirmed to work; they are educated guesses from one debugging session.
- **`Data/score_model/models/sele_score_net_d500.pt`** is a model trained on the diversified MATLAB dataset (100k curves, parameter ranges D∈[5,500], S∈[1e2,1e9], τ∈[1e-10,1e-7], 30% of samples drawn from a stratified dip-bias subset). The d32 model is older and trained on a narrower curve set.
- **The MATLAB training-data generator** lives in `MATLAB SELE Simulation/`, with `create_training_set.m` as the entry point and `calc_Sp2_single_wavelength.m` as the per-curve solver. The generator outputs `sele_simulated_*.csv` which is then ingested by the Colab training notebook.
- **The Jupyter training notebook** lives under `standalones/model_training/`. It trains the score model and produces a `.pt` checkpoint with `{model_state_dict, config, data_min, data_max}`.

What works now: load a curve from the same dataset the model was trained on, run the solver, get something that looks roughly right at the peak but is fragile to hyperparameters and shows artifacts at the surface and tail.

What does not work: anything that requires the method to generalize to a curve not from the analytical generator, or to a different material's G matrix, or to hands-off operation without per-run hyperparameter tuning.

---

## Things that were *not* tried

These are listed so you know the search space is not exhausted, in case a future debugging question reopens this codepath. None of them are recommended as next steps — the recommended next step is to switch tracks per the `AI/` direction docs.

- **Score-aware preconditioning.** Currently the score path is unpreconditioned (we just multiply by a scalar `REG_WEIGHT`). One could in principle apply a preconditioner to the score too, but the score is not the gradient of `||GS-B||²` so the same `H_inv` is not the right object — you'd need a different one, and constructing it requires assumptions about the prior we don't have.
- **Annealed `T0`.** We use a fixed small `T0` for the score evaluation. A more principled approach would anneal `T0` from large to small over the iterations (analogous to a diffusion sampler). We didn't try this.
- **Different LR schedules.** Cosine annealing was the default; we never compared against constant LR, step LR, or backtracking line search.
- **Restarting on stagnation.** When the early-stop trips on `|ΔMSE| < STOP_CHANGE`, we just stop. We never tried restarting with a perturbed `S` to escape the local basin.
- **Posterior sampling instead of MAP.** The current method finds a single point estimate (a MAP-ish solution). A diffusion posterior sampling (DPS) approach would draw samples from `p(S | B)` instead, giving uncertainty for free. This is part of the ML direction in `AI/ML_DIRECTION.md` but was not tried in this code.
- **Discrepancy-principle weight selection.** We never tied `REG_WEIGHT` to the measurement noise level. This is the biggest single missing piece for making the solver hands-free.
- **A noise model on B.** The data term is `||GS - B||²` with implicit unit weight on every wavelength. Nothing in the code uses a per-wavelength uncertainty.

---

## The minimal "make it run" recipe

If you need to reproduce the current state right now (e.g. to take a screenshot, generate a baseline, or debug a related component):

1. `uv sync`
2. Verify `Data/score_model/models/sele_score_net_d500.pt` exists (pull via DVC if not).
3. Verify `Data/score_model/G_score_model_500.csv` and `Data/score_model/datasets/sele_simulated_1000_curves_500_long_more_dip.csv` exist.
4. From repo root: `python -m src.regularization.score_model.standalones.hyperparameter_playground`
5. Expect: a single random curve reconstruction with debug prints + two plots (SELE vs GT, ELE reconstruction). Expect it to look mediocre — that is the current state, not a regression.

To reproduce the full pipeline against `MODEL_SCORE_GRAD` mode:
1. Set `CONFIG.regularization_method = "MODEL_SCORE_GRAD"` in `src/types/config.py`.
2. `python -m src.main`

---

## Where to actually go from here

Read, in order:

1. `AI/GENERIC_GUIDELINES.md` — why we are stepping away from this codepath and what the broader goal is.
2. `AI/TV_DIRECTION.md` — the recommended next direction (Total Variation + discrepancy principle, no learning).
3. `AI/ML_DIRECTION.md` — the only-if-needed direction (universal-dictionary learned prior + Plug-and-Play ADMM).

The summary of the recommendation is: this folder is a research artifact. Keep it. Don't extend it. Build the TV baseline first. Only return to a learned-prior approach if TV provably fails on multi-layer test cases — and if you do return, do not return *here*; build a fresh solver around PnP-ADMM or DPS, not around the NAG-with-score-gradient structure that lives in `score_model_grad.py`.
