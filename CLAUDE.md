# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. It contains durable domain knowledge (physics, math, architecture) distilled from the papers and codebase. It intentionally points to source files for values that change often (hyperparameters, dataset paths, exact solver steps) rather than duplicating them.

This principle applies to every README in this repo too: prefer describing concepts and pointing to
where specifics live in code over restating specifics that will drift.

### Comments

Don't overdo comments. Write them only where the WHY is genuinely non-obvious (a physics
convention, a workaround, a non-obvious ordering requirement) -- never to restate WHAT a
self-explanatory line already says. This applies to MATLAB and Python code alike.
Address the user as "sir".

### Maintaining this file

Keep CLAUDE.md a generic rulebook. When adding or editing a rule here, don't illustrate it
with an example drawn from the task you were just doing -- write the rule so it stands on
its own for any future task. Task-specific detail belongs in the PR/commit, not here.

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

### Physics discussions

When discussing physics (mechanisms, models, why a result looks the way it does), answer in
plain language first:

- Short bold-labelled paragraphs, one idea each: what we see, where it comes from, why, what
  is still open, what to check next (if relevant to discussion, you may deviate from the example format).
- Explain mechanisms in words (what a carrier or photon does), not equations. Add an
  equation or code reference only when the point depends on it.
- No tables of file paths or line numbers unless asked; say which source a claim comes from
  in a few words.
- State clearly what is confirmed vs. what is a hypothesis.
- If relevant, end with the single most useful next check, and ask before running anything numeric.

Give the full technical breakdown (equations, code locations, derivations) only when asked to
elaborate or go into detail. For a full lesson on one concept, use `/teach`.

These instructions are relevant when you're just talking to the user. When the user needs rigor,
you should follow the Research Integrity section's rules for calculations.

### Research integrity (supervisor's AI guidelines)

The user owns every equation, assumption, result, figure and conclusion. The AI speeds up the
work; it does not stand in for the user's judgment. Act so that the user can explain, check
and defend everything you produce.

**Models and assumptions.** Propose equations, models, approximations and boundary conditions;
never adopt one silently. When you introduce or change one, name it, state its assumptions
and where it comes from, and ask before building on it.

**Implementing calculations.** When code implements an equation, say which equation it is
and how its terms map to the code, including units, sign conventions and boundary
conditions. Don't call a calculation done until it has been checked against a known result:
the paper and SI (`Papers/`), the MATLAB simulator (`MATLAB SELE Simulation/`), the ground-truth test
set (`Data/test_set/`), or the group's earlier results. If no such result exists, check limiting
cases, analytic solutions, conservation laws or mesh convergence. Report which checks ran
and which did not.

**Method documentation.** Existing methods are documented loosely in `Papers/` and `plans/`.
A new method gets a Markdown file in `docs/` describing its equations, assumptions and
boundary conditions, written so it can later be converted to LaTeX for the thesis (but keep langauge simple).
When a method's equations or assumptions change, update its `docs/` file to match.

**Claims and citations.** Cite only sources you have actually read in this session or that the user
gave you. Never cite from memory, and say "unverified" when you are unsure. Separate what the
code or data shows from what is a hypothesis. A claim's only support should never be that the
AI said it.

**Research prose** (paper, thesis, reports, slides; not code docs, READMEs or plans). Write at
most one paragraph per request. Before drafting, get the key message, the supporting evidence
and the references from the user. Ask for them if they are missing rather than filling them in
yourself.

**Figures and analysis.** Any figure meant for a report, talk or paper must be produced by a
committed script in the repo, never by a scratchpad script or a one-off command. Look at
every figure you generate before reporting it, and check axis labels, units and scales
against the data. Present results; leave the interpretation and conclusions to the user,
or mark yours clearly as a suggestion.

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

### Two Absorption Coefficients

The paper distinguishes `alpha`, which includes free-carrier absorption (the Drude term),
from `alpha_b`, which excludes it. FCA attenuates the beam but is an intra-band transition
that frees no carrier, so attenuation follows `alpha` while generation and emission follow
`alpha_b` -- optical generation carries the prefactor `alpha_b / alpha` (SI eq. S4.1). This is
what makes G a _generation_ matrix rather than an absorption matrix; a row of G sums to
`alpha_b/alpha`, not to 1.

`src/optical_constants.py` is the single source of optics for the project — it reads the
paper's own ellipsometry constants from `Data/Tamir_paper_SELE_figs/`, and every G, in the
solver and in the test set alike, is built from it. Pass `k_bulk` alongside `k` when building
G; omitting it silently models absorption instead of generation.

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

---

## Commands

### Run the main pipeline

```bash
python -m src.main
```

### Score-model standalones

```bash
python -m src.regularization.score_model.standalones.hyperparameter_playground   # one config, one curve
python -m src.regularization.score_model.standalones.tune_hyperparameters        # grid search
```

Training and checkpoint-inspection scripts: see `src/regularization/score_model/README.md`.

### Rebuild the ground-truth test set

```bash
python -m src.test_set.build_test_set   # re-extract profiles from the paper figures
python -m src.test_set.plot_test_set    # overview figure of both parameter sweeps
python -m src.evaluation.plot_method_comparison   # every method against every curve
```

### Drift-diffusion p-n junction solver

```bash
python -m src.drift_diffusion.standalones.check_drift_diffusion   # physics/numerics checks
python -m src.drift_diffusion.standalones.plot_band_diagram       # band diagrams, J-V
```

### Install dependencies

```bash
uv sync
```

Key packages: `numpy`, `scipy`, `matplotlib`, `torch`, `cvxpy`, `mplcursors`, `seaborn`, `pandas`

---

## Architecture

### Entry point & flow

`src/main.py` → `src/pipeline.py:run_regularization()` — orchestrates data loading, G-matrix computation, solving, and result export.

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

**MODEL_SCORE_GRAD**: Replaces L with a learned prior from a score-based diffusion model. Score network learns `grad(log p(S))` from synthetic SELE training data. Architecture, solver, normalization and checkpoint format:
`src/regularization/score_model/README.md`.

### Pipeline flow

For the Tikhonov modes, `run_regularization()` builds G, whitens it by its median row norm
(the system is badly conditioned), sweeps κ, picks the L-curve knee (normalized distance to
the origin in log-log residual vs. seminorm), and reports the mean and std of the solutions in
a confidence window around it. The score-model branch is described in its README.

### Key modules

| File                                   | Role                                                                                                                                         |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/main.py`                          | Entry point                                                                                                                                  |
| `src/pipeline.py`                      | Orchestrates all regularization methods                                                                                                      |
| `src/types/config.py`                  | Global CONFIG, ModelScoreGradConfig, presets                                                                                                 |
| `src/__init__.py`                      | CONFIG import (avoid circular imports)                                                                                                       |
| `src/mesh.py`                          | Builds G matrix (Beer-Lambert), linear and non-uniform meshes                                                                                |
| `src/optical_constants.py`             | The project's optics: n and k with/without the Drude term                                                                                    |
| `src/matlab_fig.py`                    | Reads MATLAB `.fig` files without MATLAB                                                                                                     |
| `src/operators.py`                     | L0/L1/L2 regularization matrices                                                                                                             |
| `src/regularization/score_model/`      | Score-prior NAG solver, ScoreNetwork, standalones and training (see its README)                                                              |
| `src/test_set/`                        | Builds and loads the ground-truth SELE test set extracted from the paper's MATLAB figures                                                    |
| `src/evaluation/`                      | Runs the reconstruction methods against the test set and compares them                                                                       |
| `src/forward_model/`                   | Python port of the MATLAB SELE simulator, and the parameter space it is defined over (see its README)                                        |
| `src/drift_diffusion/`                 | Steady-state 1-D Poisson + continuity solver for a GaAs p-n junction under bias and arbitrary G(x), with band-diagram plots (see its README) |
| `src/regularization/parametric_model/` | Infers the simulator's five parameters from an ELE measurement and reconstructs SELE with uncertainty (see `plans/parametric-model.md`)      |

### Data files

| Category                      | Location                              | Contents                                                                                                                                             |
| ----------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Measurement                   | `Data/ELE_sim.csv`                    | Simulated ELE measurement (the "observation")                                                                                                        |
| Ground truth                  | `Data/SELE_ground_truth.csv`          | Ground truth SELE for comparison                                                                                                                     |
| Ground-truth test set         | `Data/test_set/`                      | Multiple ground-truth SELE profiles, each with its G and ELE, for scoring a method across shapes rather than at one operating point (see its README) |
| Paper figures                 | `Data/Tamir_paper_SELE_figs/`         | Source `.fig` files the test set and its optical constants come from (see its README)                                                                |
| Optical inputs                | `Data/z.csv`, `wavelength_nm.csv`     | z-mesh and the measurement wavelengths. k(λ) does **not** live here — it comes from the paper figures via `src/optical_constants.py`                 |
| Precomputed G matrices        | `Data/score_model/G_score_model*.csv` | One per score-model mesh resolution (target_length)                                                                                                  |
| Score model checkpoints       | `Data/score_model/models/`            | `.pt` files, one per trained score net                                                                                                               |
| Score model training datasets | `Data/score_model/datasets/`          | Synthetic SELE curve sets used to train checkpoints                                                                                                  |

`Data/score_model/models/` and `Data/score_model/datasets/` are gitignored and tracked via DVC
(`.dvc` files) rather than committed directly — pull them with DVC, not git, if missing.

### Ground-truth test set

Built by `src/test_set/` from the paper's `.fig` files; read it through
`src/test_set/loader.py`, never by parsing `index.csv` positionally. Durable points:

- **Every curve is stored on two meshes.** Its own (from the source figure, spanning the full
  wafer) and the linear solver mesh. Each has its own G and ELE, and the two ELEs differ by
  the discretization gap between them — which of the two is "the observation" is a real
  choice, not an implementation detail. `Data/test_set/README.md` quantifies it.
- **A G is data only when storing it beats recomputing it.** It is a pure function of the
  optical constants and the mesh, so an operator too large to be worth committing is left out
  of the set and rebuilt on load. Go through `loader.load_native_G` rather than reaching for
  a file, and never commit a large derived matrix when its inputs are already in the repo.
- **Resample across meshes by depth, never by index.** The profiles span the full 350 µm
  wafer while the solver mesh spans `W`; index-based resampling silently compresses a profile
  onto the wrong depths.
- **Truncating at `W` is free for the forward model but not for SELE-domain metrics.**
  Incident light is fully absorbed within a few µm, so the deep tail contributes nothing to
  the ELE — yet a large fraction of SELE _area_ lives out there. An error metric integrated
  over depth and one computed from the residual will disagree for that reason alone.
- **A coarse solver mesh cannot represent the surface value.** Its first element centre sits
  deeper than the figures' first sample, so `sele[0]` there is interpolated, not SELE(z=0) —
  relevant because SRV is read off exactly that point.
- **The set and the solver share one forward model.** Both build G from
  `src/optical_constants.py`, so the test set's `solver_mesh/G_500.csv` is bit-identical to
  what `calc_mesh_and_G` produces. Keep it that way — a benchmark whose G differs from the
  solver's measures discretization error, not reconstruction error.
- **It holds no noise.** That is deliberate, not a gap to fill unprompted, and it flatters
  every method compared against it in `src/evaluation/`. `results/` predates the optics
  unification and is not comparable to fresh output.

---

## Plotting Conventions

All plots (`src/plotting.py` and elsewhere) must be colorblind-accessible: use a colorblind-safe
categorical palette (e.g. Okabe-Ito) instead of default/arbitrary colors for distinct series, and
perceptually-uniform colorblind-safe sequential/diverging colormaps (e.g. `viridis`) for continuous data.
Don't rely on hue alone to distinguish series if avoidable — vary marker/line style too.

---

## Reference Papers

In `Papers/` directory:

- **Main paper**: "Mapping Losses through Empirical Extraction of the Spatial External Luminescence Efficiency" (Yeshurun, Fiegenbaum-Raz, Segev, ACS Appl. Energy Mater. 2024)
- **Supporting Information**: Derivations for PL calibration, optical constants, regularization method (L-curve), photon recycling model, finite element simulation details
- **PhD thesis** (Yeshurun, TAU): the full treatment, including PN junctions and MOS capacitors

Key equations: (2) forward model, (4) ELE from SELE, (5) simulated SELE via perturbation, (8) Tikhonov minimization, (11) photovoltage buildup from SELE.

Before reading any of them, read `Papers/README.md`: the built-in PDF reader does not work in
this environment, and the README gives the working extraction commands, each document's
structure, and the printed-to-PDF page offset for the thesis.
