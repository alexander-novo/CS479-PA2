#include "sample.h"

void getSamples(DataSet set, std::array<sample, CLASSES>& samples, unsigned seed) {
	std::array<unsigned, CLASSES> sizes    = getSizes(set);
	std::array<observation, CLASSES> means = getMeans(set);
	std::array<CovMatrix, CLASSES> vars    = getVars(set);
	for (unsigned i = 0; i < CLASSES; i++) {
		samples[i].resize(sizes[i]);
		genGaussianSample<DIM>(means[i], vars[i], samples[i], seed);
	}
}

std::array<observation, CLASSES> getMeans(DataSet set) {
	switch (set) {
		case DataSet::A:
			return {{{1, 1}, {4, 4}}};
		case DataSet::B:
			return {{{1, 1}, {4, 4}}};
		default:
			return {{{0, 0}, {0, 0}}};
	}
}

std::array<CovMatrix, CLASSES> getVars(DataSet set) {
	switch (set) {
		case DataSet::A:
			return {CovMatrix::Identity(), CovMatrix::Identity()};
		case DataSet::B:
			return {CovMatrix::Identity(), Vec<DIM>({4, 8}).asDiagonal()};
		default:
			return {{{1, 1}, {1, 1}}};
	}
}

std::array<unsigned, CLASSES> getSizes(DataSet set) {
	switch (set) {
		case DataSet::A:
			return {60'000, 140'000};
		case DataSet::B:
			return {40'000, 160'000};
		default:
			return {100'000, 100'000};
	}
}

std::string dataSetName(DataSet set) {
	using namespace std::string_literals;
	switch (set) {
		case DataSet::A:
			return "A"s;
		case DataSet::B:
			return "B"s;
		default:
			return "Undefined"s;
	}
}