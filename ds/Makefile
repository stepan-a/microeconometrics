TEXFILES := $(wildcard *.tex)
PDFS     := $(TEXFILES:.tex=.pdf)

all: $(PDFS)

%.pdf: %.tex
	latexmk -pdf -interaction=nonstopmode -quiet $<
	latexmk -c $<
	rm -f *.synctex.gz

clean:
	latexmk -c

distclean:
	latexmk -C

.PHONY: clean distclean
