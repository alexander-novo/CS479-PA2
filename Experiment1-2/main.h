// Part1-Bayes/main.h
#include <Eigen/Core>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>

#include "../Common/sample.h"

#define PDF_SAMPLES 100
#define ERROR_BOUND_SAMPLES 101
#define ERROR_BOUND_MIN_EPSILON 1e-10
#define ERROR_BOUND_MAX_ITERS 10

using Eigen::PartialPivLU;

// Struct for inputting arguments from command line
struct Arguments {
	DataSet set;
	unsigned seed             = 1;
	unsigned trainingSeed     = 1;
	unsigned discriminantCase = 0;
	double samplePercent      = 1;
	std::ofstream plotFiles[CLASSES], misclassPlotFiles[CLASSES], boundaryParamsFile, pdfPlotFile;
};

array<observation, CLASSES> getSampleMeans(const array<sample, CLASSES>& samples,
                                           const array<unsigned, CLASSES>& trainingSizes);
array<CovMatrix, CLASSES> getSampleVars(const array<sample, CLASSES>& samples,
                                        const array<observation, CLASSES>& sampleMeans,
                                        const array<unsigned, CLASSES>& trainingSizes);
double discriminateCase1(const observation& obs, const Vec<CLASSES>& mu, const CovMatrix& varInverse, double logVarDet,
                         double logPrior);
double discriminateCase2(const observation& obs, const Vec<CLASSES>& mu, const CovMatrix& varInverse, double logVarDet,
                         double logPrior);
double discriminateCase3(const observation& obs, const Vec<CLASSES>& mu, const CovMatrix& varInverse, double logVarDet,
                         double logPrior);
unsigned detectCase(const array<CovMatrix, CLASSES>& vars);
void calcInversesAndDets(unsigned discriminantCase, const array<CovMatrix, CLASSES>& vars,
                         array<CovMatrix, CLASSES>& varInverses, array<double, CLASSES>& varDets,
                         bool alwaysCalcDets = false);
unsigned classifySample(unsigned discriminantCase, const sample& samp, unsigned correctClass, sample& misclass,
                        observation& min, observation& max, const array<observation, CLASSES>& means,
                        const array<CovMatrix, CLASSES>& varInverses, const array<double, CLASSES>& logVarDets,
                        const array<double, CLASSES>& logPriors, bool plotMisclassifications = false);
void printPlotFile(std::ofstream& plotFile, const sample& samp);
double printPdfPlotFile(std::ofstream& pdfPlotFile, const observation& min, const observation& max,
                        const array<observation, CLASSES>& means, const array<CovMatrix, CLASSES>& varInverses,
                        const array<double, CLASSES>& varDets, const array<double, CLASSES>& priors);
void printParamsFile(std::ofstream& boundaryParamsFile, const observation& min, const observation& max,
                     const array<observation, CLASSES>& means, const array<CovMatrix, CLASSES>& varInverses,
                     const array<double, CLASSES>& logVarDets, const array<double, CLASSES>& logPriors, double pdfMax);
bool verifyArguments(int argc, char** argv, Arguments& arg, int& err);
void printHelp();

double (*const discriminantFuncs[])(const observation&, const Vec<CLASSES>&, const CovMatrix&, double, double) = {
    discriminateCase1, discriminateCase2, discriminateCase3};

// Taken from https://stackoverflow.com/a/9345144/4530832
template <class BidiIter>
BidiIter random_unique(BidiIter begin, BidiIter end, size_t num_random, unsigned seed) {
	size_t left = std::distance(begin, end);

	std::mt19937_64 engine(seed);
	while (num_random--) {
		BidiIter r = begin;
		std::uniform_int_distribution<unsigned> dist(0, left);
		std::advance(r, dist(engine));
		std::swap(*begin, *r);
		++begin;
		--left;
	}
	return begin;
}