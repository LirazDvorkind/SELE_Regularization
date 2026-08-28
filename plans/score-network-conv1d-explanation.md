# Why the score network becomes convolutional

Background and design rationale for the change specified in
[`score-network-conv1d-plan.md`](score-network-conv1d-plan.md).

This document is written to be readable without deep neural-network background. It explains
what is wrong, how we know, and why each element of the replacement architecture was chosen.
It deliberately explains concepts rather than restating code — for the exact current network
see `src/regularization/score_model/model_definition.py`.

---

## 1. The problem

The d500 score model generates SELE curves whose *mean shape* is right but which are visibly
jittery. Quantified with a roughness metric — rms of the second difference divided by the
standard deviation, i.e. "how much does this wiggle relative to its own size":

| | roughness |
|---|---|
| Real training curves | ~0.0003 |
| Generated samples | ~0.9 |

A factor of roughly **3000**.

## 2. What it is not

Two obvious explanations were tested and ruled out by measurement.

**It is not the sampler.** The deterministic probability-flow ODE gives roughness 0.99, versus
0.92 for Euler–Maruyama. Removing the stochastic noise injection makes it slightly *worse*.
Whatever produces the jitter is present in the network's own output, not introduced by the
integration scheme.

**It is not undertraining or insufficient capacity.** An *exact* Tweedie denoiser was built
from held-out curves — a ground-truth reference for what a perfect model of this dataset would
predict. The network matches its RMSE at every noise level and beats it below `t = 0.05`. Its
score magnitudes are also well calibrated (23.53 vs 23.30 at `t = 0.5`; 489.96 vs 500.88 at
`t = 0.01`).

So the network *has* learned the distribution. But the ideal denoiser's roughness is
0.0003–0.0005 while the network's is 0.49–1.76.

**The conclusion those two measurements force:** the residual error is small in amplitude but
almost entirely high-frequency. The network is accurate and rough at the same time.

## 3. The cause

`ScoreNetwork` is a **multilayer perceptron (MLP)** — a stack of fully-connected layers. Its
500 output values are each an independently-weighted sum of all 628 inputs. Output #200 and
output #201 are produced by two entirely separate sets of weights that share nothing.

The cleanest way to see what this means: take the whole dataset and shuffle the depth axis —
move depth 400 to where depth 3 was, the same shuffle for every curve — then apply that same
shuffle to the network's weights. **The network behaves identically.** Same accuracy, same
training curve.

Depth ordering carries *zero* architectural information. The network knows only "500 slots."

Smoothness — the fact that a SELE curve does not jump between neighbouring depths — is
therefore something it must infer statistically from examples, rather than something it gets
for free. It half-learned it: enough to get the shape right, not enough to suppress per-point
static.

This is the same gap Song & Ermon's 2019 NCSN closes for images by being fully convolutional:
no layer ever predicts a pixel in isolation, so errors emerge spatially correlated rather than
as per-pixel noise. The 1D fix is the same fix.

**Intended outcome of the change: generation produces smooth curves on its own**, with no
smoothing, filtering, or projection applied at sampling time.

---

## 4. Reading a tensor shape

Every arrow in a network carries a block of numbers; the shape gives its dimensions. Read it
as a stack of tables.

`(B, 128, 500)` means:

- **B** — how many curves are processed at once (the batch). Pure throughput, no architectural
  meaning. Mentally set it to 1.
- **128** — **channels**. At each depth point the network tracks 128 numbers instead of one.
- **500** — **length**. The 500 depth points of the SELE curve.

So for one curve, `(1, 128, 500)` is a 128 × 500 grid: 500 columns (one per depth), 128 rows
(one per feature).

**Why 128 numbers per depth point?** The raw curve has one number per depth — the SELE value,
shape `(1, 1, 500)`. The network expands that to 128 internal features: think of 128 different
questions asked at every depth ("is the curve rising here", "is there a peak", "how curved is
it"). The network invents its own features; it is never given labels like those. The 128
numbers at depth 200 are its *description* of what is happening around depth 200. At the end a
head collapses them back to one number per depth — the answer.

**The MLP by contrast carries `(B, 628)`.** Two dimensions, no length axis at all: a flat list
with no structure, where position 200 and 201 are unrelated slots. That missing dimension is
the entire problem.

---

## 5. The design choices

### 5.1 Convolution instead of fully-connected

A convolution is a small sliding window — here 5 depth points wide — with **one shared set of
weights**, dragged along the entire curve. The same local pattern detector runs at depth 3, at
depth 200, at depth 480.

Two consequences, both the point of the exercise:

- **Locality becomes structural.** Every output is computed from a neighbourhood, never from
  the whole curve at once. Independent per-point jitter becomes hard to express rather than
  being the default.
- **Weight sharing collapses the parameter count.** One window of 5 weights replaces 500
  independent rows.

*Sources:* Song & Ermon 2019 (NCSN) is fully convolutional; DiffWave uses "a stack of residual
layers with bidirectional dilated convolution, kernel size 3" for 1D audio.

### 5.2 Dilation, to reach across the curve

A 5-wide window sees 5 points. Stacking them naively grows reach 5, 9, 13, 17 — roughly 125
layers to span a 500-point curve, which is unaffordable.

**Dilation spreads the window's taps apart**: look at every 2nd point, then every 4th, every
8th. With kernel size 5 the reach is `1 + 4·Σdᵢ`, so dilations (1, 2, 4, 8, 16, 32, 64) reach
**509 points in seven layers**. Each layer zooms out one octave. Local detail is kept early,
global structure arrives by the end.

**Bidirectional, not causal.** WaveNet and DiffWave use causal convolutions for audio because
time has a direction. Depth does not — what happens at 300 µm is legitimately informed by what
happens at 200 µm and at 400 µm. The window is centred.

### 5.3 A dilated stack, not a U-Net

The standard image-diffusion shape is a U-Net, which downsamples then upsamples. That imposes
divisibility constraints on the input length, and this codebase must serve both
`target_length = 500` and `32`. A dilated stack needs no resampling and the same code covers
both — at L=32 the dilations simply cap at (1, 2, 4, 8), reach 61.

### 5.4 Residual (skip) connections

Each block **adds** its result to its input rather than replacing it. Standard since ResNet:
gradients flow through deep stacks, and each block learns a small correction rather than a
whole transformation. The current MLP already does this; it carries over unchanged.

### 5.5 GroupNorm

**What a normalization layer is:** as numbers flow through successive layers they drift toward
very large or very small values and training destabilizes. A normalization layer rescales them
back to a standard mean and spread. The variants differ in *what they average over*:

| | averages over | note |
|---|---|---|
| BatchNorm | the batch | unreliable at small batch sizes; largely abandoned in diffusion |
| LayerNorm | all features of one sample | what the current MLP uses |
| GroupNorm | groups of channels within one sample (e.g. 8 groups of 16) | batch-size independent; the diffusion-convnet standard (Dhariwal & Nichol's ADM) |

**⚠️ Flagged risk.** GroupNorm normalizes across the length axis, which strips per-sample
*amplitude* out of the activations — and amplitude is the dominant variance direction in this
dataset (PCA: 4 components carry 99.99% of variance, the first essentially being scale). The
residual skips should carry it through. **If retrained samples come out with correct shapes but
collapsed amplitude spread, drop the norm layers entirely** — the network is small enough not
to need them.

### 5.6 SiLU instead of Softplus

The activation is the nonlinearity between layers; without one the whole stack collapses to a
single linear map. Softplus is a smooth ReLU, a 2010s choice. **SiLU** (`x·sigmoid(x)`, also
called swish) is the modern diffusion default — smooth, permits small negative values,
empirically better gradient behaviour. Low stakes, but free.

### 5.7 FiLM time conditioning — the one that is not optional

**The problem it solves.** The network must do a different job depending on how noisy its
input is. At high noise (`t` near 1) the input is nearly pure static and it should make broad,
coarse guesses; at low noise (`t` near 0) the input is nearly a real curve and it should make
fine corrections. Same weights, different behaviour, selected by `t`.

The current MLP glues `t` onto the input as 128 extra numbers and hopes the network sorts it
out. **In a convolutional network that is meaningless**, because a convolution slides along
the depth axis and `t` does not live at a depth. Placing it at "position 501" says nothing.

**FiLM** (Feature-wise Linear Modulation) does it properly, and is simpler than the name: a
small side-network turns `t` into two numbers per channel — a multiplier **γ** and an offset
**β** — and inside every block each channel is scaled by its γ and shifted by its β. One line
of arithmetic: `h = γ·h + β`.

#### Worked example, 4 channels instead of 128

Activations at some point in the network (4 channels × 5 depths):

```
              depth→   d1    d2    d3    d4    d5
channel 1 (edges)     0.2   0.9   0.4  -0.1   0.3
channel 2 (peaks)     1.1   0.8   1.4   0.6   0.2
channel 3 (curvature) 0.0  -0.3   0.5   0.7  -0.2
channel 4 (level)     2.0   2.1   1.9   2.2   2.0
```

FiLM takes `t = 0.8` (high noise), runs it through a tiny MLP, and emits:

```
          γ      β
ch 1     0.1    0.0     ← at high noise, ignore edge detail
ch 2     0.3    0.0     ← damp peak detection
ch 3     0.0    0.0     ← curvature is useless in static: switch off
ch 4     1.5    0.2     ← amplify overall level, this still matters
```

Applied row by row — every depth in a row gets the same γ and β:

```
              depth→   d1    d2    d3    d4    d5
channel 1            0.02  0.09  0.04 -0.01  0.03    (squashed)
channel 2            0.33  0.24  0.42  0.18  0.06    (damped)
channel 3            0.00  0.00  0.00  0.00  0.00    (off)
channel 4            3.20  3.35  3.05  3.50  3.20    (boosted)
```

At `t = 0.05` the side-network emits a completely different γ/β set, presumably reviving
channels 1 and 3 now that fine detail matters. **`t` acts as a bank of volume knobs, one per
feature, re-set at every block.**

#### The two properties that make it correct here

1. **γ and β are per-channel, not per-depth** — shape `(B, 128, 1)`, the `1` broadcasting
   across all 500 depths. This is right because noise level is a property of the whole sample,
   uniform across depth. It would be wrong to let `t` say something different at depth 3 than
   at depth 400.
2. **It is injected into every block, not once at the input.** In the MLP, `t` enters at the
   front door and is progressively diluted through six layers. ADM, DiffWave and Stable
   Diffusion's U-Net all condition this way rather than by concatenation.

### 5.8 Replicate padding, not zeros

A sliding window falling off the end of the curve needs values beyond the boundary. The library
default invents **zeros**, which would teach the network that SELE goes to zero at `z = 0`.

That is physically wrong: **surface SELE is nonzero, and it is precisely the quantity SRV is
read from** (see the SELE-shape table in `CLAUDE.md`). Replicate padding repeats the edge value
instead. This choice is domain reasoning specific to this project, not a convention inherited
from a paper.

### 5.9 Zero-initialize the final output conv

Standard diffusion practice. The network begins by predicting exactly zero score, so early
training does not have to fight large random outputs.

---

## 6. Side by side

| | Current (MLP) | Proposed (dilated conv) |
|---|---|---|
| How an output is computed | own weighted sum of *all* inputs | local window, weights shared across depth |
| Knows depth ordering | **no** | yes, structurally |
| Reach across the curve | global immediately | 509 points via 7 dilated layers |
| Time conditioning | concatenated once at input | FiLM into every block |
| Norm / activation | LayerNorm + Softplus | GroupNorm + SiLU |
| Boundaries | n/a | replicate padding |
| Widest tensor | `(B, 2048)` = 2,048 values | `(B, 128, 500)` = 64,000 values |
| Largest single weight | `2048×2048` = 4.2 M | `128×128×5` = 82 K |
| Parameters | 15.61 M | ≈1.7 M |
| Checkpoint | 62 MB | ≈7 MB |
| Activation memory | small | ~9× larger |

### Layer trace — current MLP

Tensors are flat, `(batch, features)`. No length axis exists anywhere.

```
x (B, 500)          t (B, 1)
                     └─ SinusoidalTimeEmbedding ──► (B, 128)
concat ──────────────────────────────────────────► (B, 628)

ResidualBlock  628 →  512    (B,  512)     0.64 M
ResidualBlock  512 → 1024    (B, 1024)     1.05 M
ResidualBlock 1024 → 2048    (B, 2048)     4.20 M
ResidualBlock 2048 → 2048    (B, 2048)     4.20 M
ResidualBlock 2048 → 1024    (B, 1024)     4.20 M
ResidualBlock 1024 →  512    (B,  512)     1.05 M
Linear         512 →  500    (B,  500)     0.26 M
                                          ───────
                                           15.61 M
```

15.61 M × 4 bytes ≈ 62.4 MB, matching the 62,439,919-byte checkpoint — so this reconstruction
is confirmed rather than estimated. Note that **the three middle layers hold 12.6 M of the
15.6 M parameters**, each a dense square matrix, and none of it buys depth adjacency.

### Layer trace — proposed conv

The tensor is 3D, and **the length axis never changes**: 500 in, 500 out throughout. Only the
channel count moves.

```
x (B, 500) ──► unsqueeze ──► (B, 1, 500)

t (B, 1) ─► Sinusoidal(128) ─► MLP 128→256→256 ─► t_emb (B, 256)   0.10 M
                                                    │
stem  Conv1d(1→128, k=5, d=1)        (B, 128, 500)  │            0.0008 M
                                                    │
Block 1  d=1    (B,128,500)   RF   13 ◄── FiLM ─────┤             0.23 M
Block 2  d=2    (B,128,500)   RF   29 ◄── FiLM ─────┤             0.23 M
Block 3  d=4    (B,128,500)   RF   61 ◄── FiLM ─────┤             0.23 M
Block 4  d=8    (B,128,500)   RF  125 ◄── FiLM ─────┤             0.23 M
Block 5  d=16   (B,128,500)   RF  253 ◄── FiLM ─────┤             0.23 M
Block 6  d=32   (B,128,500)   RF  509 ◄── FiLM ─────┤  ← full curve
Block 7  d=64   (B,128,500)   RF 1021 ◄── FiLM ─────┘             0.23 M

head  GroupNorm + Conv1d(128→1, k=1, zero-init)  (B, 1, 500)     0.0001 M
squeeze ─────────────────────────────────────────► (B, 500)
                                                                 ───────
                                                                  ≈1.7 M
```

Inside one residual block:

```
in                          (B, 128, 500)
├─ Conv1d(128→128, k=5, d)  (B, 128, 500)    82,048   ← replicate pad, 2d per side
├─ GroupNorm(8 groups)      (B, 128, 500)       256
├─ SiLU                     (B, 128, 500)         0
├─ FiLM: t_emb → (γ, β)     (B,256) → 2×(B,128,1)  65,792
│      h = γ·h + β          broadcast across all 500 depths
├─ Conv1d(128→128, k=5, d)  (B, 128, 500)    82,048
├─ GroupNorm(8 groups)      (B, 128, 500)       256
├─ SiLU
└─ + in  (identity skip)    (B, 128, 500)
                                            ─────────
                                              230,400
```

**Three consequences worth knowing before training:**

- **Fewer weights does not mean faster.** ~9× smaller in parameters but substantially *more*
  activation memory — 128 channels × 500 length is a far bigger intermediate than a flat 2048
  vector. Expect to lower the Colab batch size, and do not expect step time to fall in
  proportion to the parameter count. It may not fall at all.
- **RF 509 at block 6 is the design target.** By then every output point has seen the whole
  curve, so global shape stays fully representable — the MLP's global reach is retained while
  gaining the locality it never had. Block 7 (RF 1021) is redundant margin and can be dropped
  for 6 blocks / ≈1.5 M.
- **The `target_length = 32` variant is the same code**, dilations capped at (1, 2, 4, 8) —
  3–4 blocks, ≈0.9 M parameters.

---

## 7. What this does not fix

This change targets **generation quality only**. It does not by itself fix the NAG solver
collapsing to the mean curve. That has three separate, already-identified causes, each a
follow-up experiment:

1. The solver queries the network without the VP mean-scale factor `a_t` (0.281 at `t = 0.5`),
   feeding an unscaled `S_norm` at time `t`. Measured effect: correct Tweedie RMSE 0.1297 vs
   solver-style 0.7524, and solver-style correlation-to-mean of **−0.542**.
2. Global min–max normalization squashes 28% of training curves into a span narrower than
   `σ_t` at the solver's own `T0 = 0.05`.
3. The adaptive norm rescaling discards the network's calibrated score magnitude.

Also deliberately out of scope: per-curve amplitude normalization, and ε-prediction instead of
the raw-score output convention.

---

## 8. Sources

- Song & Ermon, *Generative Modeling by Estimating Gradients of the Data Distribution*
  (NCSN, 2019) — fully convolutional score networks; spatially correlated error structure.
- Kong et al., *DiffWave* (2021) — 1D diffusion with stacked bidirectional dilated
  convolutions; the closest published analogue to this problem.
- van den Oord et al., *WaveNet* (2016) — the dilation-cycle construction and receptive-field
  arithmetic.
- Dhariwal & Nichol, *Diffusion Models Beat GANs* (ADM, 2021) — GroupNorm + SiLU blocks,
  per-block time conditioning, zero-initialized output layers.
- Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer* (2018) — the
  scale-and-shift conditioning mechanism.
