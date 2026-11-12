# JMLR manuscript

This folder retains both the original JMLR manuscript and the revised,
self-contained article. From the repository root, run `make jmlr-original`
or `make jmlr-revised`; `make` builds both together with the two parallel
manuscripts. These targets keep temporary files in `out/` and copy the final
PDF beside its source entry point.

`main.tex` is the main entry point. Build from this directory with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

From the repository root, use `latexmk -cd -pdf jmlr-final/main.tex`.

- `preamble.tex`: packages, theorem environments, and shared macros.
- `frontmatter.tex`: title, authors, editor, and title generation.
- `abstract.tex`: abstract.
- `sections/`: main-text sections named `section1.tex`, `section2.tex`, and so on.
- `tables/`: comparison table.
- `appendices/`: one file per appendix, named `appendixA.tex`, `appendixB.tex`, and so on.
- `ref.bib`: bibliography for this manuscript; edit this file rather than `main.bbl`.
- `jmlr2e.sty`: local JMLR style dependency.

`main.tex` and its modular section files are the original JMLR manuscript and
are intentionally preserved.  The current revised, self-contained article
version is `revised_main.tex`; it uses `revised_header.tex` and
`revised_ref.bib`, with its comparison table in
`tables/revised-ncsc-complexity.tex`, and can be built from this
directory with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error revised_main.tex
```

The corresponding compiled output is `revised_main.pdf`.

The former standalone root template has been removed from the project; the
original JMLR entry point remains `main.tex` in this directory.

The revised article includes `numerical_experiments.tex` and
`numerical_experiments_appendix.tex`. Their static PDF figures remain in
`figures/experiments/`, so compiling this folder does not require the former
top-level experiment code. The obsolete figure-regeneration script was
archived with that code during repository cleanup.

Build verification on 2026-10-28: both entry points compile. The preserved
original manuscript still has three unresolved references:
`sec:experiments`, `sec:conclusion`, and `app:parameter-free-scope`.
The revised entry point has no unresolved references.
