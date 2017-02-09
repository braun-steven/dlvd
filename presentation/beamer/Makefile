PAPER = pres 
TEX = $(wildcard *.tex)
BIB = Master.bib

.PHONY: all clean

$(PAPER).pdf: $(TEX) $(BIB) 
	pdflatex $(PAPER)
	biber $(PAPER)
	pdflatex $(PAPER)
	pdflatex $(PAPER)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out $(PAPER).pdf

