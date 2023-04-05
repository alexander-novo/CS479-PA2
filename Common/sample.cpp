#include "sample.h"

#include <algorithm>

void getSamples(DataSet set, std::array<sample, CLASSES>& samples, unsigned seed) {
	std::array<unsigned, CLASSES> sizes    = getSizes(set);
	std::array<observation, CLASSES> means = getMeans(set);
	std::array<CovMatrix, CLASSES> vars    = getVars(set);
	for (unsigned i = 0; i < CLASSES; i++) {
		samples[i].resize(sizes[i]);
		genGaussianSample<DIM>(means[i], vars[i], samples[i], seed);
	}
}

void getSamplesFromFile(std::ifstream& in_file, array<sample, CLASSES>& samples) {
	observation temp;
	unsigned class_idx;

	while (true) {
		for (unsigned i = 0; i < DIM; i++) { in_file >> temp[i]; }

		in_file >> class_idx;

		if (!in_file) return;

		samples[class_idx].push_back(temp);
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

observation sampleMean(sample::const_iterator begin, sample::const_iterator end) {
	observation mean  = observation::Zero();
	unsigned size     = std::distance(begin, end);
	double correction = 1.0 / size;

#pragma omp parallel for reduction(+ : mean)
	for (unsigned i = 0; i < size; i++) { mean += correction * *(begin + i); }

	return mean;
}

CovMatrix sampleVariance(sample::const_iterator begin, sample::const_iterator end, const observation& sampleMean) {
	CovMatrix var     = CovMatrix::Zero();
	unsigned size     = std::distance(begin, end);
	double correction = 1.0 / (size - 1);

#pragma omp parallel for reduction(+ : var)
	for (unsigned i = 0; i < size; i++) {
		observation temp = (*(begin + i) - sampleMean);
		var += temp * temp.transpose();
	}

	return correction * var;
}