# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. It contains durable domain knowledge (physics, math, architecture) distilled from the papers and codebase. It intentionally points to source files for values that change often (hyperparameters, dataset paths, exact solver steps) rather than duplicating them.

This principle applies to every README in this repo too: prefer describing concepts and pointing to
where specifics live in code over restating specifics that will drift.

### Comments

Don't overdo comments. Write them only where the WHY is genuinely non-obvious (a physics
convention, a workaround, a non-obvious ordering requirement) -- never to restate WHAT a
self-explanatory line already says. This applies to MATLAB and Python code alike.

### Maintaining this file

Keep CLAUDE.md a generic rulebook. When adding or editing a rule here, don't illustrate it
with an example drawn from the task you were just doing -- write the rule so it stands on
its own for any future task. Task-specific detail belongs in the PR/commit, not here.

### Docs stand alone

Every README covers its own folder and nothing else. It never links to another README, a
plan, or this file, and never depends on one to make sense: when it needs a fact owned
elsewhere, it says it in a line of its own words or names the code that holds it. Naming code
paths is fine; naming other docs is not. A little repetition is the price of independence --
pay it rather than create a chain of "see X". Project-wide physics and conventions live in
this file only, and folder-specific detail lives only in that folder's README, so this file
names folders but does not restate them. Plans in `plans/` are working notes: docs never
point to them.

### Writing files from the shell

Don't write multi-line file content through a Bash heredoc (`cat > file <<'EOF'`). On this
Windows/Git-Bash setup, long heredocs containing apostrophes, backticks, or quotes fail to
parse with `unexpected EOF while looking for matching`, and the file is left unwritten. Use
the Write tool (or Edit) for file content; keep Bash for commands. If a script must be run
from a temporary file, Write it first, then run it.

### Commit messages

One short sentence that captures the gist of the change without glossing over anything
important in it. No bullet points. A multi-sentence body is for a genuinely large change
that cannot be summarized honestly in one line -- prefer the minimum.

### Branches

Creating, deleting, renaming, merging, rebasing, resetting, or pushing to a branch -- any
change to a branch other than committing to the one already checked out -- needs the user's
explicit permission first, every time. Permission for one branch operation does not cover
the next. Worktrees count: don't create one, or the branch it needs, without asking.

### Tone

These rules shape only the final wording of replies, never reasoning or work products.
Drop the voice if asked.

Speak as a calm, dry British aide and address the user as "sir", with at most one light
remark per reply. Accuracy and honesty about uncertainty come first; point out flaws politely.

Lead with the answer, in a few plain sentences unless detail is needed. Explain any jargon.
No preambles, filler or self-praise. Write prose, without bold "Label. text" paragraphs or
headings in short replies. If the user doesn't understand, try a more concrete angle.

### Physics discussions

Answer in plain prose, in the tone above: what we see, where it comes from and why.
Explain mechanisms in words (what a carrier or photon does), not equations, and add an
equation or code reference only when the point depends on it. No tables of file paths or
line numbers unless asked; name a claim's source in a few words. State clearly what is
confirmed and what is a hypothesis. End with the single most useful next check, and ask
before running anything numeric.

Give the full technical breakdown (equations, code locations, derivations) only when asked to
elaborate or go into detail. For a full lesson on one concept, use `/teach`.

---

## Project Overview

This project solves an **inverse problem** in semiconductor photovoltaics. Given wavelength-dependent External Luminescence Efficiency (ELE) measurements of a GaAs wafer, it reconstructs the **Spatial External Luminescence Efficiency (SELE)** -- a depth profile showing how much each layer contributes to photoluminescence.

### The Forward Model

```
eta_ext(lambda_in) = (1 / phi_abs(lambda_in)) * integral_0^d  DeltaG(lambda_in, z) * S(z) dz
```

- `eta_ext`: measured ELE per wavelength
- `DeltaG(lambda, z)`: Beer-Lambert optical generation: `alpha = 4*pi*k/lambda`, `G[lambda,i] = exp(-alpha*z_i) - exp(-alpha*z_{i+1})`
- `S(z)`: the SELE profile to reconstruct (the unknown)
- `phi_abs`: absorbed photon flux

Discretized: `eta_ext = (1/phi_abs) * G @ S`, where G is (L wavelengths × M depth elements), S is (M,). The system is ill-posed -- small noise causes wild oscillations in naive least-squares solutions. Regularization is required.

Because `phi_abs = phi_0 * A` already normalizes by the absorbed fraction, front-surface
reflectance cancels out of the discretized system and must not be applied to G a second time.

Incident light is absorbed within a few µm, so the data say nothing about deeper SELE: the
deep tail contributes nothing to the ELE, yet holds much of the SELE's area. An error metric
integrated over depth and one computed from the ELE residual will disagree for that reason
alone.

### Two Absorption Coefficients

The paper distinguishes `alpha`, which includes free-carrier absorption (the Drude term),
from `alpha_b`, which excludes it. FCA attenuates the beam but is an intra-band transition
that frees no carrier, so attenuation follows `alpha` while generation and emission follow
`alpha_b` -- optical generation carries the prefactor `alpha_b / alpha` (SI eq. S4.1). This is
what makes G a _generation_ matrix rather than an absorption matrix; a row of G sums to
`alpha_b/alpha`, not to 1.

`src/optical_constants.py` is the single source of optics for the project — it reads the
paper's own ellipsometry constants from `Data/Tamir_paper_SELE_figs/`, and every G, in the
solver, the simulator and the test sets alike, is built from it. Pass `k_bulk` alongside `k`
when building G; omitting it silently models absorption instead of generation. Keep every G
builder on this one source: a benchmark whose G differs from the solver's measures
discretization error, not reconstruction error.

### Physical Intuition for the SELE Profile

Expected shape for a p-type GaAs wafer (paper Figure 2b):

- **Near surface (z ~ 0)**: LOW (~0.05-0.1 %) due to surface recombination suppressing radiative recombination
- **Rises to peak** (~0.3-0.45 %) at ~0.5-1 µm; peak position correlates with minority carrier diffusion length
- **Decays toward 0** deeper (self-absorption); the profile approaches ~0 at the back of the 350 µm wafer

| Physical Parameter                   | Effect on SELE                                                                                |
| ------------------------------------ | --------------------------------------------------------------------------------------------- |
| Surface recombination velocity (SRV) | Controls dip at z=0. Higher SRV = lower surface SELE.                                         |
| SRH lifetime (tau_SRH)               | Controls bulk SELE magnitude. Longer lifetime = higher bulk SELE.                             |
| Diffusion length                     | Peak position ~ diffusion length when surface recomb dominates.                               |
| Photon recycling                     | Increases quasi-Fermi level separation deep in material; keeps SELE nonzero far from surface. |

### Units

Depth in cm and SELE as a fraction, everywhere in code and data. Paper figures plot µm and
percent; convert on the way in.

---

## Commands

```bash
python -m src.main                                 # the main pipeline, mode set in src/types/config.py
python -m src.evaluation.plot_method_comparison    # every method against every test curve
python -m src.evaluation.feasibility_demo          # interactive: what the ELE data can and cannot pin down
```

Every other runnable script is listed in the README of the folder it lives in.

---

## Architecture

### Entry point & flow

`src/main.py` → `src/pipeline.py:run_regularization()` — orchestrates data loading, G-matrix computation, solving, and result export to `results/pipeline/`.

### Regularization modes (set via `CONFIG.regularization_method`)

`src/regularization/parametric_model/` is a fourth approach that does not go through this
enum or the pipeline: it infers the simulator's parameters rather than regularising a curve,
and is driven by its own standalones. The three below are the curve-space methods.

| Mode               | Solver                                | Key file                  |
| ------------------ | ------------------------------------- | ------------------------- |
| `NON_UNIFORM_MESH` | Classical Tikhonov with adaptive mesh | `tikhonov_non_uniform.py` |
| `TOTAL_VARIATION`  | Two-parameter Tikhonov (CVXPY)        | `total_variation.py`      |
| `MODEL_SCORE_GRAD` | Nesterov gradient + score prior       | `score_model_grad.py`     |

**NON_UNIFORM_MESH**: `min ||G@S - B||² + κ²||L@S||²`, L = second-derivative operator. Sweeps κ, finds L-curve knee, averages solutions in confidence window. Non-uniform mesh: fine linear spacing near surface, logarithmic deeper. This is the original paper's method.

**TOTAL_VARIATION**: Two-parameter Tikhonov solved with CVXPY. Allows sharper features than standard Tikhonov.

**MODEL_SCORE_GRAD**: Replaces L with a learned prior from a score-based diffusion model. Score network learns `grad(log p(S))` from synthetic SELE training data.

The Tikhonov modes whiten G by its median row norm (the system is badly conditioned) and pick
the L-curve knee by normalized distance to the origin in log-log residual vs. seminorm.

### Where things live

| Path                                   | Role                                                                                         |
| -------------------------------------- | -------------------------------------------------------------------------------------------- |
| `src/__init__.py`                      | CONFIG import (avoid circular imports)                                                       |
| `src/optical_constants.py`             | The one source of optics                                                                     |
| `src/regularization/`                  | The three pipeline solvers, plus `score_model/` and `parametric_model/`                      |
| `src/forward_model/`                   | Python port of the MATLAB SELE simulator and its five-parameter space                        |
| `src/test_set/`                        | Builds and loads both ground-truth test sets                                                 |
| `src/evaluation/`                      | Compares the methods against the test set; the feasibility demo                              |
| `src/pn_junction/`                     | Drift-diffusion solver, junction SELE, and a synthetic junction measurement                  |
| `Data/test_set/`                       | SELE profiles from the paper's figures, each with its G and ELE                              |
| `Data/test_set_forward_model/`         | SELE profiles made by `src/forward_model/` itself, so no model mismatch                      |
| `Data/Tamir_paper_SELE_figs/`          | The paper's `.fig` files: test-set profiles and the optical constants                        |
| `Data/score_model/`, `Data/parametric_model/` | Checkpoints and training sets, tracked by DVC, not git — pull them with DVC          |
| `MATLAB SELE Simulation/`              | The original MATLAB simulator                                                                |
| `Papers/`                              | The paper, its SI, the thesis, and the follow-up manuscript                                  |
| `results/`                             | Generated output, one folder per script                                                      |

Most folders carry a README for their own detail. Read the test sets through
`src/test_set/loader.py`, never by parsing their `index.csv` positionally.

---

## Plotting Conventions

All plots (`src/plotting.py` and elsewhere) must be colorblind-accessible: use a colorblind-safe
categorical palette (e.g. Okabe-Ito) instead of default/arbitrary colors for distinct series, and
perceptually-uniform colorblind-safe sequential/diverging colormaps (e.g. `viridis`) for continuous data.
Don't rely on hue alone to distinguish series if avoidable — vary marker/line style too.

---

## Reference Papers

The built-in PDF reader does not work in this environment. Before reading anything in
`Papers/`, read `Papers/README.md`: it gives the working extraction commands, each document's
structure, and the printed-to-PDF page offset for the thesis.
