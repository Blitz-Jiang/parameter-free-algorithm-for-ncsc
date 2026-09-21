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

References to unwritten sections or proofs remain unresolved; no placeholder sections have been added.

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
