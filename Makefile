CXXFLAGS     = -std=c++14 -g -fopenmp -O3 -D_GLIBCXX_PARALLEL
OBJDIR       = obj
DEPDIR       = $(OBJDIR)/.deps
# Flags which, when added to gcc/g++, will auto-generate dependency files
DEPFLAGS     = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d

# Function which takes a list of words and returns a list of unique words in that list
# https://stackoverflow.com/questions/16144115/makefile-remove-duplicate-words-without-sorting
uniq         = $(if $1,$(firstword $1) $(call uniq,$(filter-out $(firstword $1),$1)))

# Source files - add more to auto-compile into .o files
SOURCES      = Common/sample.cpp Common/image.cpp Experiment1-2/main.cpp Experiment3/main.cpp
INCLUDES     = -I include/
# Executable targets - add more to auto-make in default 'all' target
EXEC         = Experiment1-2/classify Experiment3/classify-skin
# Targets required for the homework, spearated by part
REQUIRED_1   = 
REQUIRED_2   = 
REQUIRED_OUT = $(REQUIRED_1) $(REQUIRED_2)

SOURCEDIRS   = $(call uniq, $(dir $(SOURCES)))
OBJDIRS      = $(addprefix $(OBJDIR)/, $(SOURCEDIRS))
DEPDIRS      = $(addprefix $(DEPDIR)/, $(SOURCEDIRS))
DEPFILES     = $(SOURCES:%.cpp=$(DEPDIR)/%.d)

.PHONY: all exec clean report out/A-table.tex out/B-table.tex
.SECONDARY:

# By default, make all executable targets and the outputs required for the homework
all: exec $(REQUIRED_OUT) Report/report.pdf
exec: $(EXEC)

# Executable Targets
Experiment1-2/classify: $(OBJDIR)/Experiment1-2/main.o $(OBJDIR)/Common/sample.o
	$(CXX) $(CXXFLAGS) $^ -o $@

Experiment3/classify-skin: $(OBJDIR)/Experiment3/main.o $(OBJDIR)/Common/image.o
	$(CXX) $(CXXFLAGS) $^ -o $@

### Experiments 1-2 Outputs ###
out/A-table.tex out/B-table.tex:
	rm -f $@
	touch $@

.SECONDEXPANSION:
out/sample1-%.dat out/sample2-%.dat out/params-%.dat out/classification-rate-%.txt: Experiment1-2/classify | out
	@Experiment1-2/classify $(word 1,$(subst -, ,$*)) -ps1 out/sample1-$(word 1,$(subst -, ,$*)).dat\
	                                                  -ps2 out/sample2-$(word 1,$(subst -, ,$*)).dat\
	                                                  -pdb out/params-$*.dat\
	                                                  -p   $(word 2,$(subst -, ,$*))\
	                                                  -t   out/$(word 1,$(subst -, ,$*))-table.tex | tee out/classification-rate-$*.txt

out/sample-plot-%.png: out/sample1-%.dat out/sample2-%.dat out/params-%.dat Experiment1-2/plot.plt
	@gnuplot -e "outfile='$@'"\
	         -e "sample1='out/sample1-$*.dat'"\
	         -e "sample2='out/sample2-$*.dat'"\
	         -e "plotTitle='Decision Boundaries on Data Set $*'"\
	         -e "paramFile='out/params-$*-.01.dat'"\
			 -e "paramFile2='out/params-$*-100.dat'"\
			 -e "percent1='.01'"\
			 -e "percent2='100'"\
	         Experiment1-2/plot.plt

### Experiment 3 Outputs ###


# Figures needed for the report
# Part 1
report: out/A-table.tex out/B-table.tex
report: out/sample1-A-.01.dat out/sample1-A-.1.dat out/sample1-A-1.dat out/sample1-A-10.dat out/sample1-A-100.dat
report: out/sample1-B-.01.dat out/sample1-B-.1.dat out/sample1-B-1.dat out/sample1-B-10.dat out/sample1-B-100.dat
report: out/sample-plot-A.png out/sample-plot-B.png

Report/report.pdf: Report/report.tex report
	latexmk -pdf -cd -use-make -silent -pdflatex='pdflatex -interaction=batchmode -synctex=1' $<

clean:
	rm -rf $(OBJDIR)
	rm -f $(EXEC)
	rm -rf out
	rm -f Images/*.png
	cd Report/; latexmk -c

# Generate .png images from .pgm images. Needed for report, since pdfLaTeX doesn't support .pgm images
%.png: %.pgm
	pnmtopng $< > $@

%.png: %.ppm
	pnmtopng $< > $@

# Auto-Build .cpp files into .o
$(OBJDIR)/%.o: %.cpp
$(OBJDIR)/%.o: %.cpp $(DEPDIR)/%.d | $(DEPDIRS) $(OBJDIRS)
	$(CXX) $(DEPFLAGS) $(INCLUDES) $(CXXFLAGS) -c $< -o $@

# Make generated directories
$(DEPDIRS) $(OBJDIRS) out: ; @mkdir -p $@
$(DEPFILES):
include $(wildcard $(DEPFILES))