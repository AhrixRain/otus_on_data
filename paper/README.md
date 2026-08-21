# paper/

Working LaTeX skeleton for the Run D / Run E paper.

- `main.tex`: full section skeleton with measured Run D numbers filled in and
  `[TODO]` markers for Run E results.
- `references.bib`: starting bibliography; expand and deduplicate before
  submission.
- `figures/`, `tables/`: place Run E PNG/PDF figures and generated tables here.

Compile:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
