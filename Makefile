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

.PHONY: all exec clean report
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


### Experiment 3 Outputs ###


# Figures needed for the report
# Part 1
report: 

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
$(DEPDIRS) $(OBJDIRS) out out/bayes out/euclid: ; @mkdir -p $@
$(DEPFILES):
include $(wildcard $(DEPFILES))