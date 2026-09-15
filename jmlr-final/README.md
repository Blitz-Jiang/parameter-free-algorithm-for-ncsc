# JMLR manuscript

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

The repository-root `jmlr_main.tex` forwards to this manuscript for compatibility. The root `main.tex`, `ref.bib`, and style file remain available to the earlier manuscript.

References to unwritten sections or proofs remain unresolved; no placeholder sections have been added.
