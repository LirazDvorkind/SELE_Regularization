# Where Tikhonov was used on a PN/PIN structure — the Soreq PIN archive (2026-10-27)

Question: Gideon said the Soreq archive holds PN simulations that solve SELE with Tikhonov.
Where is that, and did the thesis use Tikhonov for junctions?

Sources read: `Soreq PIN 20240718.zip` (Google Drive, `Thesis/`; extracted to a scratch folder, not
into the repo), and the thesis (`Papers/`, §2.4 and §3.4). The zip's `.mat` and `.fig` files were
read as data only.

## 1. The thesis does not use Tikhonov for the junction

§3.4 (PDF pp. 70–80) gets the thin-film GaAs cell's SELE by forward simulation (COMSOL, TMM
optics, perturb-and-measure). It never inverts a measured ELE for a junction. Tikhonov appears
only for wafers (§2.4, §3.2) and the MOS capacitor (§3.5.3, wide-range variant, L0 operator,
linear mesh 0–1.5 µm).

## 2. What the archive is

It is not a simulation. It is SELE extraction from a **measured PLQY** of the Soreq PIN sample.

| file | role |
|---|---|
| `Extract_SELE_GaAs_PIN_20240718.m` | main script: 390–500 nm, 100 elements, 0.5 µm domain |
| `..._good.m` | same, 390–650 nm, 50 elements, 1.5 µm domain |
| `..._test_short_wavlengths.m` | same values as `_good`, plots commented out, reopens a `.fig` |
| `Extract_SELE_InP_20250507.m` | same pipeline on n-type InP; reflectance from a file; 30 elements |
| `Combined_PLQY_soreq1.mat` | the measurement: 64 points, 400–695 nm, PLQY 1.4e-5 to 1.3e-4 |
| `n_k_GaAs_w_Drude_150323.mat` | GaAs optical constants with the Drude term |
| `*.fig` (3) | SELE curves at many mesh resolutions and domain lengths |

The `.mat` carries no layer stack or sample description, so the device is not identifiable
from the archive alone.

## 3. The method in the scripts

Same as the thesis wafer method; nothing junction-specific.

- Forward model: Beer–Lambert generation matrix on a linear mesh, applied to measured PLQY.
- Regulariser: L0 (magnitude only), via `cgsvd` + `tikhonov`.
- κ swept over 501 values (1e-5 to 1e-12); L-curve corner via `l_corner`.
- Result: mean of the solutions in a window around the knee; forward-projected back to PLQY
  as a check.

## 4. Differences from this repo's forward model (to check, not yet resolved)

- **Reflectance:** the scripts multiply G by `(1-R)`. CLAUDE.md says reflectance cancels against
  `phi_abs` and must not be applied to G again. Unclear whether the script's data are
  normalised the same way.
- **No `alpha_b/alpha` prefactor:** the script uses a single `alpha` (with the Drude term), so it
  models absorption, not generation (SI eq. S4.1).
- **Optics:** plain Beer–Lambert, no TMM, although the sample is a layered device.
- **Depth:** the domain is 0.5–1.5 µm.

## 5. What the `.fig` files show (data, not interpretation)

Each overlays extractions at several `x_res` and domain lengths. Units match the PLQY, so peaks
are about 3e-4 (0.03%).

- The maximum's position is not stable against the mesh. On the 1.5 µm domain it sits at about
  0.04–0.09 µm for `x_res` 27–60, jumps to about 0.43–0.45 µm for 75–200, is 0.31 µm at 300
  and 0.10 µm at 800.
- Many curves go negative, worse for finer meshes on shorter domains. On the 0.3 µm domain,
  `x_res`=250 spans about −1.4e-3 to +1.1e-3, larger than the PLQY scale.
- The maximum also moves with domain length (about 0.18–0.2 µm at 0.5 µm, 0.45 µm at 1 µm).

This matches the thesis's own warning that the mesh "must be designed with great attention"
and that the element count is chosen as the smallest giving non-negative SELE.

## 6. Open

- No curve in the `.fig` files is marked as the reported result. Which `x_res` and domain
  were used in the final result is unknown; ask Gideon or Tamir.
- Hypothesis, untested: the unstable peak position means the 390–500 nm data constrain the
  first fraction of a micron poorly.
- Next check, if wanted: run the 390–500 nm case through this repo's solver on this PLQY and
  see whether the same mesh sensitivity appears.
