LATEXMK ?= latexmk
TEXFLAGS := -pdf -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all fixed-radius jmlr-original jmlr-revised second-order clean
.NOTPARALLEL:

all: fixed-radius jmlr-original jmlr-revised second-order

fixed-radius:
	$(LATEXMK) -cd $(TEXFLAGS) -outdir=out fixed-radius-newton/main.tex
	cp fixed-radius-newton/out/main.pdf fixed-radius-newton/main.pdf

jmlr-original:
	$(LATEXMK) -cd $(TEXFLAGS) -outdir=out jmlr-final/main.tex
	cp jmlr-final/out/main.pdf jmlr-final/main.pdf

jmlr-revised:
	$(LATEXMK) -cd $(TEXFLAGS) -outdir=out jmlr-final/revised_main.tex
	cp jmlr-final/out/revised_main.pdf jmlr-final/revised_main.pdf

second-order:
	$(LATEXMK) -cd $(TEXFLAGS) universal-sharp-utr/main.tex
	$(LATEXMK) -cd -c -bibtex- universal-sharp-utr/main.tex

clean:
	rm -rf fixed-radius-newton/out jmlr-final/out universal-sharp-utr/out
	$(LATEXMK) -cd -c -bibtex- universal-sharp-utr/main.tex
