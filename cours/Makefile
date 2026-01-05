TEXFILES := $(wildcard *.tex)
PDFS     := $(TEXFILES:.tex=.pdf)

all: $(PDFS)

%.pdf: %.tex
	@rubber --src-specials --unsafe --pdf $<
	@rubber --clean $<

clean-all:
	@rm -f *.pdf

.PHONY: all
