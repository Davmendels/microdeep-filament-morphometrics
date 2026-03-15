# Microdeep Filament Morphometrics
## ZORTH
### Research Utility

Lightweight Python pipeline for morphometric analysis of filamentous microorganisms
(e.g. *Arthrospira* / Spirulina).

The method estimates filament length and diameter from contour geometry
using perimeter–area inversion.

## Repository structure

code/
    Python measurement pipeline

data/
    example microscopy images

notes/
    measurement note (Pandoc → PDF)

## Build documentation

pandoc notes/morphometrics.md \
  --template=notes/template/template.tex \
  --pdf-engine=lualatex \
  -o notes/morphometrics.pdf

## Citation

If this code is used in scientific work, please cite:

Microdeep Filament Morphometrics – ZORTH Technologies (2026)

or reference the Microdeep system where appropriate.
