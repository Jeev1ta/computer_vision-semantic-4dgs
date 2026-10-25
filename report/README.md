# IEEE-style project report

- `ieee_project_report.tex` — IEEE conference-format LaTeX source.
- `references.bib` — BibTeX bibliography.
- `workflow.dot` — editable Graphviz source for the code-path diagram.
- `workflow_diagram.pdf` and `workflow_diagram.png` — generated diagram outputs.

## Build a PDF

Install a LaTeX distribution that includes `IEEEtran.cls` and BibTeX (for example, TeX Live with the recommended LaTeX packages), then run from this directory:

```bash
pdflatex ieee_project_report.tex
bibtex ieee_project_report
pdflatex ieee_project_report.tex
pdflatex ieee_project_report.tex
```

Build from this directory so `workflow_diagram.pdf` and the `../assets/` figure path resolve. This workspace currently has no `pdflatex`, `latexmk`, or `bibtex` executable available, so the manuscript has not been compiled to PDF here. The qualitative RGB/semantic image is an existing repository visualization asset whose checkpoint provenance is unavailable; it is labeled as illustrative, not as a measured result. The full test suite also needs PyTorch and the official Gaussian-Splatting submodule; the currently available CPU subset passes 41 tests but has one test blocked by that missing upstream loader. The report intentionally omits unverified benchmark scores and marks scene-level ablations as not run. Before submission, verify author/affiliation details, perform the experiments, replace the illustrative image with outputs from a documented checkpoint and held-out data, add the results, and compile/check the final PDF against the required IEEE template and page limits.
