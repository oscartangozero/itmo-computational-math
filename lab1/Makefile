.PHONY: all report source clean

all: report source
report: out/report.pdf
source: out/src.zip

out:
	mkdir out

out/src.zip: $(wildcard src/*.py)
	zip -r $@ $^

out/%.pdf: %.tex out
	latexmk -pdf -shell-escape -outdir=out $<

subfiles = $(filter-out report.tex,examples.tex $(wildcard *.tex))
report.tex: $(subfiles) examples.tex

examples.tex:
	./generate_examples.py examples.tex test/basic/*

clean:
	rm -rf out examples.tex
