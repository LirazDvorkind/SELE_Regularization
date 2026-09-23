# Papers

Reference material for the project. Read this file first when asked to consult a paper.

## What is here

| file | what it is | pages |
|---|---|---|
| `Mapping Losses through Empirical Extraction of the Spatial External Luminescence Efficiency.pdf` | The main paper (Yeshurun, Fiegenbaum-Raz, Segev, ACS Appl. Energy Mater. 2024). GaAs wafer SELE, Tikhonov/L-curve extraction. | 7 |
| `SELE Mapping Supporting Information.pdf` | Its SI: PL calibration, optical constants, regularization details, photon-recycling model, FEM setup. Pages are numbered S-1.. | 23 |
| `Ph_D__Thesis___SELE___Tel_Aviv_University___Yeshurun.pdf` | The full thesis. Everything above plus InP wafers, PN junctions, MOS capacitors, operando oxide growth, and appendices on instruments and error analysis. | 131 |
| `An analythic solution to the SELE of a GaAs wafer - for Zohara.docx` | Closed-form SELE derivation for the wafer case — the precursor that became section 3 of the manuscript below, plus two sections it dropped (`Relating Sp to S`, `Obtaining results by fitting to the ratio of two spectra`). 7 review comments. | — |
| `Zohara-Full_document_rev4-GS.docx` | **The follow-up manuscript**, under review. Fits the closed-form PL expression to excitation–emission PL maps to extract diffusion length, and defines an SRV ranking index. 64 review comments, mostly Gideon Segev. | — |

## How to read the PDFs

The project venv has no PDF library and `pdftoppm` is not installed, so the built-in PDF
reader does not work here. Use PyMuPDF through `uv run --with` (temporary, does not touch
`pyproject.toml`), run from the repo root:

```bash
# text of a page range (0-based indices; here PDF pages 70-80)
uv run --with pymupdf python -c "
import pymupdf; d = pymupdf.open('Papers/<file>.pdf')
for i in range(69, 80): print(f'===== PDF PAGE {i+1} =====\n' + d[i].get_text())"

# render pages to PNG to look at figures (then open the PNG with the Read tool)
uv run --with pymupdf python -c "
import pymupdf; d = pymupdf.open('Papers/<file>.pdf')
for i in range(69, 80): d[i].get_pixmap(dpi=100).save(f'<scratchpad>/p{i+1}.png')"

# bookmarks / table of contents with PDF page numbers
uv run --with pymupdf python -c "
import pymupdf; [print(l, t, p) for l, t, p in pymupdf.open('Papers/<file>.pdf').get_toc()]"
```

Write scripts longer than a few lines to a file first (see the CLAUDE.md rule on heredocs).
Put rendered PNGs and text dumps in the session scratchpad, not in this directory.

## How to read the `.docx` files

`pandoc` is **not installed** here, so the usual `pandoc -t markdown` route fails. A `.docx`
is a ZIP of XML — unzip it and parse with stdlib `ElementTree` (`lxml` is not installed
either). Two traps:

- **Set `PYTHONIOENCODING=utf-8`.** Console output defaults to cp1252 and these documents
  carry en-dashes, Greek and LTR marks; without it the dump dies mid-way on `UnicodeEncodeError`.
- **Equations are not in `<w:t>`.** They are OMML (`<m:oMath>`) in the `math` namespace, so a
  dump that only walks `w:t` silently renders every equation as an empty line followed by its
  number. Convert `m:f`, `m:sSub`, `m:sSup`, `m:nary`, `m:d` and `m:rad` recursively to get
  readable linear form.

Comments live in `word/comments.xml` (author, date, body, keyed by id); their anchors in the
body text are `<w:commentRangeStart/End>` markers, so interleave those into the document dump
to see what each comment is attached to. Tracked changes are `<w:ins>` / `<w:del>`, and inside
a `<w:del>` the text element is `<w:delText>`, not `<w:t>`.

The `anthropic-skills:docx` skill covers editing and comment authoring; for read-only work the
above is enough. Keep dump scripts and their output in the scratchpad.

## Thesis navigation

**Printed page = PDF page − 23.** The user quotes printed page numbers; the front matter
(abstract, nomenclature, lists) is 23 PDF pages. The bookmarks for the appendices are
broken (they point back into the front matter) — locate appendix sections by text search.

Main body, by PDF page:

| section | PDF pages |
|---|---|
| Introduction: PL, SCE (eq 1.1–1.3), ELE (eq 1.4–1.6), SELE (eq 1.7–1.9) | 24–33 |
| Methodology: PL/ELE setup and calibration | 34–39 |
| Optical constants, TMM, generation profile | 40–41 |
| SELE extraction methods (Tikhonov variants, projected GD, inverse Laplace) | 44–48 |
| SELE simulation, PL simulation, photon recycling model | 49–55 |
| Samples: GaAs and InP wafers, native oxides, MOS capacitors | 56–59 |
| Results: SELE of a GaAs wafer (the main paper's content) | 60–66 |
| Results: SELE of an InP wafer | 67–69 |
| Results: SELE of PN junctions (thin-film GaAs cell, bias, materials) | 70–80 |
| Results: MOS capacitor surface characterization | 81–89 |
| Results: operando native-oxide SELE | 90–91 |
| Results summary, conclusions | 92–94 |
| Appendices (instruments, error analysis, calibration, SOPs) | ~95–131 |

Figures that matter most for this repo: the wafer SELE and its parameter sweeps (PDF 60–66,
the source of `Data/test_set/`); optical constants (PDF 40, the source of
`src/optical_constants.py`); the PN-junction SELE and SCE (Fig 3.10–3.11, PDF 73–75, discussed
in `plans/pn-junction-sele-primer.md`).

## Main paper and SI

Neither has bookmarks. The main paper's key equations are numbered (2) forward model,
(4) ELE from SELE, (5) simulated SELE by perturbation, (8) Tikhonov, (11) photovoltage from
SELE. The SI is sectioned S1–S6; eq S4.1 is the `alpha_b/alpha` generation prefactor that
turns an absorption matrix into a generation matrix.

That prefactor is applied by every G builder in the project — `src/mesh.py` (which can omit
it, and then models absorption instead), `src/test_set/build_test_set.py`, and
`src/forward_model/analytic_ele.py`. All three take `k` and `k_bulk` from
`src/optical_constants.py`, which is the single source of optics and reads the paper's own
ellipsometry out of `Data/Tamir_paper_SELE_figs/`. Go there for the constants rather than to
any one consumer of them.

## The follow-up manuscript and this repo's forward model

`Zohara-Full_document_rev4-GS.docx` goes in a different direction from the solver: instead of
inverting `eta_ext = (1/phi_abs)·G·S` numerically, it fits a closed-form `PL(lambda_em,
lambda_ex)` and leaves only the diffusion length and one transport parameter as unknowns. Its
equations are nonetheless the *same* equations as `MATLAB SELE Simulation/calc_Sp2.m` and
`src/forward_model/` — the first MATLAB commit implemented that model term for term.

Before comparing the two, read `plans/forward-model-drift-from-zohara-equations.md`. It maps
each manuscript equation to its code counterpart, states where the repo deliberately differs
and what each side assumes, and lists issues in the manuscript that the existing review comments do not catch. Two things in it will save re-deriving:

- **The manuscript's `A₁` is the code's `A2`/`a2`.** The code's `A1` is the literal constant
  `2` — the manuscript's leading `2` in its Eq (11). Everyone maps this wrong once.
- **The repo's `G` carries `alpha_b/alpha`; the manuscript's Eq (14) does not.** Once that is
  accounted for, the repo's closed-form ELE agrees with the manuscript's Eq (16) to
  discretisation error.
