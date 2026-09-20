# Papers

Reference material for the project. Read this file first when asked to consult a paper.

## What is here

| file | what it is | pages |
|---|---|---|
| `Mapping Losses through Empirical Extraction of the Spatial External Luminescence Efficiency.pdf` | The main paper (Yeshurun, Fiegenbaum-Raz, Segev, ACS Appl. Energy Mater. 2024). GaAs wafer SELE, Tikhonov/L-curve extraction. | 7 |
| `SELE Mapping Supporting Information.pdf` | Its SI: PL calibration, optical constants, regularization details, photon-recycling model, FEM setup. Pages are numbered S-1.. | 23 |
| `Ph_D__Thesis___SELE___Tel_Aviv_University___Yeshurun.pdf` | The full thesis. Everything above plus InP wafers, PN junctions, MOS capacitors, operando oxide growth, and appendices on instruments and error analysis. | 131 |
| `An analythic solution to the SELE of a GaAs wafer - for Zohara.docx` | Closed-form SELE derivation for the wafer case. | — |

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
`src/mesh.py` applies.
