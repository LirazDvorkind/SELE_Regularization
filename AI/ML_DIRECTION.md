# Direction B — Universal-Dictionary Learned Prior + Plug-and-Play ADMM

This is the direction to try **only if `TV_DIRECTION.md` proves insufficient**. Read `GENERIC_GUIDELINES.md` first for the broader goal, and read `TV_DIRECTION.md` for the baseline this needs to beat.

The core question this direction answers is: *if a hand-crafted regularizer (TV) is not flexible enough, can we learn a regularizer that is generic across materials, without needing per-material training data?*

---

## The core idea — and why it differs from what we tried

What we tried before: train a score model on synthetic GaAs SELE curves from one analytical generator, then use it as a prior on GaAs reconstruction. This is **not** generic. It's a one-material model dressed up as a learned prior, and the moment we point it at a different material it has nothing useful to say.

The new idea has two ingredients that change everything:

### Ingredient 1 — A universal SELE dictionary

Instead of training on curves from one material's analytical generator, train on a deliberately-diverse dictionary of SELE curves designed to span the space of *all* SELE curves we might ever see, regardless of material. Concretely, the dictionary should contain:

- **Single-layer curves** with widely varied diffusion lengths, surface recombination velocities, and bulk lifetimes. (We already do this for GaAs; just sweep the parameters even more aggressively.)
- **Two-layer stacks** with random interface depths and random per-layer parameters. The interface should sometimes be a sharp jump, sometimes a smooth crossover, sometimes barely visible.
- **Three-layer stacks** for the same reasons.
- **Curves with no underlying physical model at all** — random smooth perturbations, random piecewise-smooth profiles, random monotone profiles. The point is to *not* let the model overfit to any one physical generator.

The training data does **not** need to correspond to physically realizable materials. The point is to teach the prior the *space of plausible SELE shapes*, not the *space of GaAs SELE shapes*. The richer the dictionary, the more material-agnostic the resulting prior.

### Ingredient 2 — Plug-and-Play ADMM (or DPS) instead of NAG-with-score

Once we have a learned prior, the question is how to combine it with the data fidelity term. Our current approach (Nesterov gradient descent on `||GS-B||² + REG_WEIGHT · score`) has no convergence guarantee, no automatic weighting, no clean separation between data and prior.

**Plug-and-Play ADMM** solves all of these at once. The structure is:

```
repeat:
    z ← argmin_S  ||GS - B||² + ρ ||S - (x - u)||²    # data step (closed form)
    x ← Denoise(z + u)                                # prior step (one forward pass of the learned model)
    u ← u + (z - x)                                   # dual update
```

The data step is a linear least-squares solve — fast, deterministic, no learning rate to tune. The prior step calls a denoiser (which is what a score model effectively *is* at a given noise level). The dual update enforces consistency between the two. The key facts:

- **It converges.** Under mild conditions on the denoiser, PnP-ADMM has a fixed point and provable monotone behavior. We are not flying blind on momentum and LR like we are now.
- **There is no `REG_WEIGHT` to tune.** The relative weight between data and prior emerges from `ρ` (the ADMM penalty parameter), and `ρ` can be chosen by standard ADMM heuristics — or, better, set automatically using the discrepancy principle on the data residual at convergence.
- **The prior is *modular*.** PnP just needs *a denoiser*. Could be our score model (if we retrain it on the universal dictionary). Could be a small U-Net. Could even be BM3D for the first prototype. The framework doesn't care, and we can swap denoisers without rewriting the solver.

An alternative to PnP-ADMM is **Diffusion Posterior Sampling (DPS)**, which uses a score model directly inside a reverse-diffusion sampler conditioned on the measurement. DPS has the advantage of producing *samples* from the posterior, which gives uncertainty quantification for free (run it N times, look at the spread). PnP-ADMM gives a point estimate. Both are valid; DPS is harder to implement well but gives better uncertainty.

---

## Why this is different from what we did before

| Aspect | What we did | What this proposes |
|---|---|---|
| Training data | 1k–100k GaAs curves from one analytical formula | Universal dictionary spanning many materials and stack geometries |
| Solver | NAG with hand-weighted score gradient | PnP-ADMM (or DPS) with provable convergence |
| Regularization weight | `REG_WEIGHT` tuned by hand per material | ADMM `ρ` set automatically; data weight from discrepancy principle |
| Validation | Same generator as training | Held-out generator(s) we never trained on |
| Uncertainty | None | Bootstrap (PnP) or posterior sampling (DPS) |
| Failure mode | Score dominates → same shape every curve | Convergence guarantee bounds the worst case |

The biggest single change is the **dictionary**. None of the other changes matter if the prior has only seen one material's curves.

---

## How to know it's actually generic

This is where we have to be honest with ourselves. The trap last time was that we tested on curves from the same generator we trained on, so "it works" meant nothing. The validation protocol for this direction has to satisfy:

1. **Train on the universal dictionary.** Single-, two-, three-layer synthetic stacks plus random non-physical curves.
2. **Test on curves from a generator we did NOT train on.** Examples:
   - Finite-element simulations of a real device (the paper's Figure 2 supplementary data).
   - Published SELE measurements from other materials (perovskite, CIGS, organic) — if we can get them.
   - Hand-crafted multi-layer profiles with deliberate features (sharp interfaces, thin layers, near-zero regions).
3. **Compare against TV+discrepancy on the same test set.** If we can't beat TV, we don't need this direction. If we beat TV on the multi-layer cases but tie on the single-layer ones, we've justified the investment.

The order is important. Step 3 is the gate. Skipping it sends us right back to "we built a thing and it looks plausible, ship it."

---

## What's hard about this direction

- **Building the universal dictionary is most of the work.** It is much more work than swapping the analytical generator's parameter ranges, and it requires designing a multi-layer SELE simulator (which we don't currently have). The whole approach hinges on this dataset being right.
- **Choosing the denoiser noise level is subtle.** PnP works best when the denoiser is calibrated to the right noise level for the current iterate, not a fixed `T0`. There are standard tricks (annealed PnP, RED) but they add complexity.
- **Verifying convergence in practice** is harder than the theory suggests because the denoiser is only approximately a proximal operator. We will need diagnostics (residual decrease, fixed-point distance) to know when something is going wrong.
- **Compute cost.** Training on a universal dictionary is much heavier than training on 1k GaAs curves. Probably needs a GPU and a few hours, not a laptop and a few minutes.
- **It can still fail to be generic.** The dictionary might miss the kind of structure that shows up in some real material. There is no formal guarantee that "diverse training data" means "generalizes to anything." We can only check empirically.

---

## When this direction is worth starting

Not yet. The honest sequence is:

1. Implement TV+discrepancy as in `TV_DIRECTION.md`.
2. Test it on synthetic single-layer (sanity check), synthetic multi-layer (the real test), and any real measurements we can get.
3. **If TV is good enough, stop here.** A learned method we can't justify is worse than a classical method we can.
4. If TV is not good enough, look at *which specific cases* it fails on. Those failures are the design specification for the universal dictionary — they tell us what kinds of curves the learned prior actually needs to know about.
5. Only then start building the dictionary, training the prior, and wiring up PnP-ADMM.

The temptation will be to skip steps 1-3 because "the ML direction is more interesting." Resist it. The lesson from the score-grad work is that it's very easy to spend months on a learned method whose failures we can't diagnose because we have no baseline to compare to. TV is the baseline. Build it first.

---

## What this *isn't*

Not a plan. Not a spec. A direction. The actual choices — which denoiser architecture, which ADMM variant, how to schedule `ρ`, how to construct the dictionary, what sampling strategy to use — should be made at implementation time, with the failure modes of TV in front of us as design input. Writing this down now is just to make sure that *if* we go in this direction, we go in for the right reasons and with the right guardrails, instead of repeating the score-grad mistakes with bigger models.
