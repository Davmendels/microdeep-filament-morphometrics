# Microdeep Filament Morphometrics
## ZORTH – Research Utility

Lightweight Python pipeline for morphometric analysis of filamentous microorganisms
(e.g. *Arthrospira* / Spirulina).

The method estimates filament **length** and **diameter** from contour geometry
using a perimeter–area inversion model.

The repository provides a complete workflow:

YOLO detection → ROI extraction → contour segmentation → morphometric estimation → CSV export.

---

## Quickstart

```bash
python3 -m venv venv-morpho
source venv-morpho/bin/activate
pip install -r code/requirements.txt

python code/detect_and_measure.py \
  --input data/images \
  --weights data/model/filaments.pt \
  --output results.csv
```

This command will:
  • detect filament-like objects using the YOLOv8 model
  • extract ROIs around detected filaments
  • perform adaptive thresholding and morphological closing
  • extract contours
  • estimate filament length and diameter
  • export morphometrics to a CSV file

---

## Example output

The generated CSV contains one row per detected filament:

```
image,det_id,class,confidence,
area_px2,perimeter_px,length_px,diameter_px,aspect_ratio,valid
```

All measurements are reported in **pixel units**.

Conversion to physical units requires microscope calibration.

---

## Repository structure

code/
    detect_and_measure.py     YOLO detection + measurement pipeline
    filament_measure.py       morphometric core functions

data/
    images/                   example microscopy images
    model/                    example YOLO filament detector

notes/
    morphometrics.md          measurement note
    morphometrics.pdf         measurement note
    template/                 Pandoc LaTeX template

The compiled PDF of the measurement note is included for convenience.
Users with Pandoc and LaTeX installed can regenerate it from the Markdown source.
---

## Build documentation

pandoc notes/morphometrics.md \
  --template=notes/template/template.tex \
  --pdf-engine=lualatex \
  -o notes/morphometrics.pdf

---

## Method overview

Filament geometry is estimated from contour area (A) and perimeter (P).

Assuming approximately constant filament diameter:

x² - (P/2)x + A = 0

Solving the quadratic equation yields two roots interpreted as:
  • filament **length**
  • filament **diameter**

Further details are provided in the accompanying measurement note.

---

## Citation

If this code is used in scientific work, please cite:

**Microdeep Filament Morphometrics – ZORTH (2026)**

or reference the **Microdeep system** where appropriate.

⸻

** License

See LICENSE file.
