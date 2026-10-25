# IEEE-style project report

- `ieee_project_report.tex` — IEEE conference-format LaTeX source.
- `references.bib` — BibTeX bibliography.

## Build a PDF

Install a LaTeX distribution that includes `IEEEtran.cls` and BibTeX (for example, TeX Live with the recommended LaTeX packages), then run from this directory:

```bash
pdflatex ieee_project_report.tex
bibtex ieee_project_report
pdflatex ieee_project_report.tex
pdflatex ieee_project_report.tex
```

This workspace currently has no `pdflatex`, `latexmk`, or `bibtex` executable available, so the source has not been compiled to PDF here. The report intentionally does not include unverified benchmark scores. Before submitting it, add the actual course/department details, verify author spellings and affiliation, run the experiments described in Section V, insert reproducible results, and compile/check the final PDF using the required IEEE template and page limits.
