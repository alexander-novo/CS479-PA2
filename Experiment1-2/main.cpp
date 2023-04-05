// Experiment1-2/main.cpp
#include "main.h"

int main(int argc, char** argv) {
	int err;
	Arguments arg;

	if (!verifyArguments(argc, argv, arg, err)) { return err; }

	sample misclassifications[CLASSES];
	array<sample, CLASSES> samples;
	array<double, CLASSES> priors, logPriors, varDets = {}, logVarDets = {};
	array<CovMatrix, CLASSES> varInverses;
	array<unsigned, CLASSES> sizes = getSizes(arg.set);

	if (arg.dataInputFile.is_open()) {
		getSamplesFromFile(arg.dataInputFile, samples);
	} else {
		getSamples(arg.set, samples, arg.seed);
	}

	std::cout << "Training on " << arg.samplePercent * 100 << "% of data\n";
	array<unsigned, CLASSES> trainingSizes;

	// Only reshuffle data if the training seed has been specifically set.
	// This way, we can compare to other implementations.
	if (arg.trainingSeed != Arguments::DEFAULT_TRAINING_SEED) {
		// Randomly shuffle samples. When we pick a subsample to train on, we just pick the first n
		// observations in each sample. So to change the subsamples, we shuffle differently.
		// Only shuffle the first n observations from among all of the observations.
		std::transform(samples.begin(), samples.end(), trainingSizes.begin(), [&arg](sample& sample) {
			random_unique(sample.begin(), sample.end(), sample.size() * arg.samplePercent, arg.trainingSeed);
			return sample.size() * arg.samplePercent;
		});
	}

	array<observation, CLASSES> means = getSampleMeans(samples, trainingSizes);
	array<CovMatrix, CLASSES> vars    = getSampleVars(samples, means, trainingSizes);

	observation min = samples[0].front(), max = samples[0].front();

	double totalSize = std::accumulate(sizes.begin(), sizes.end(), 0);

	// Compute priors and their logs
	std::transform(sizes.cbegin(), sizes.cend(), priors.begin(), logPriors.begin(),
	               [totalSize](unsigned size, double& prior) {
		               prior = size / totalSize;
		               return log(prior);
	               });

	std::cout << "Classifying data set \"" << dataSetName(arg.set) << "\" - " << CLASSES << " classes.\n";

	for (unsigned i = 0; i < CLASSES; i++) {
		std::cout << "Class " << (i + 1) << ":\n"
		          << "Sample Mean:\n"
		          << means[i] << "\nSample Variance:\n"
		          << vars[i] << "\n\n";
	}

	// Compute which case we're in from the book
	if (arg.discriminantCase == 0) {
		arg.discriminantCase = detectCase(vars);

		std::cout << "Detected case " << arg.discriminantCase << "\n\n";
	} else {
		std::cout << "Overriden case " << arg.discriminantCase << "\n\n";
	}

	std::cout << "Prior probabilities:\n";
	std::for_each(priors.begin(), priors.end(), [](double prior) { std::cout << prior << '\n'; });
	std::cout << '\n';

	calcInversesAndDets(arg.discriminantCase, vars, varInverses, varDets);

	// Calculate the logs of the determinants
	std::transform(varDets.cbegin(), varDets.cend(), logVarDets.begin(), [](const double& det) { return log(det); });

	unsigned overallMisclass = 0;

	std::cout << "Misclassification rates:\n";
	for (unsigned i = 0; i < CLASSES; i++) {
		sample& misclass = misclassifications[i];

		unsigned misclassCount = classifySample(arg.discriminantCase, samples[i], i, misclassifications[i], min, max,
		                                        means, varInverses, logVarDets, logPriors);

		std::cout << misclassCount / (double) sizes[i] << "\n";

		overallMisclass += misclassCount;

		if (arg.plotFiles[i]) { printPlotFile(arg.plotFiles[i], samples[i]); }
	}

	std::cout << "\nOverall misclassification rate:\n" << overallMisclass / totalSize << "\n\n";

	if (arg.boundaryParamsFile) {
		printParamsFile(arg.boundaryParamsFile, min, max, means, varInverses, logVarDets, logPriors);
	}

	if (arg.tabularDataFile) {
		array<observation, CLASSES> realMeans = getMeans(arg.set);
		array<CovMatrix, CLASSES> realVars    = getVars(arg.set);

		arg.tabularDataFile << arg.samplePercent * 100 << "\\% & ";

		for (unsigned i = 0; i < CLASSES; i++) {
			arg.tabularDataFile << (realMeans[i] - means[i]).norm() / realMeans[i].norm() << " & ";
		}
		for (unsigned i = 0; i < CLASSES; i++) {
			arg.tabularDataFile << (realVars[i] - vars[i]).norm() / realVars[i].norm() << " & ";
		}

		arg.tabularDataFile << (overallMisclass / totalSize) << "\\\\\n";
	}

	return 0;
}

array<observation, CLASSES> getSampleMeans(const array<sample, CLASSES>& samples,
                                           const array<unsigned, CLASSES>& trainingSizes) {
	array<observation, CLASSES> means;

	for (unsigned i = 0; i < CLASSES; i++) {
		means[i] = sampleMean(samples[i].begin(), samples[i].begin() + trainingSizes[i]);
	}

	return means;
}

array<CovMatrix, CLASSES> getSampleVars(const array<sample, CLASSES>& samples,
                                        const array<observation, CLASSES>& sampleMeans,
                                        const array<unsigned, CLASSES>& trainingSizes) {
	array<CovMatrix, CLASSES> vars;

	for (unsigned i = 0; i < CLASSES; i++) {
		vars[i] = sampleVariance(samples[i].begin(), samples[i].begin() + trainingSizes[i], sampleMeans[i]);
	}

	return vars;
}

#pragma region Discriminant functions
double discriminateCase1(const observation& obs, const Vec<CLASSES>& mu, const CovMatrix& varInverse, double logVarDet,
                         double logPrior) {
	// Note that varInverse is what is passed, and the inverse of a diagonal matrix is also a
	// diagonal matrix with the diagonal elements being the reciprocal of the original elements. So
	// instead of dividing by sigma^2, we multiply by 1/sigma^2.
	return (varInverse(0, 0) * mu).dot(obs) - varInverse(0, 0) * mu.dot(mu) / 2.0 + logPrior;
}

double discriminateCase2(const observation& obs, const Vec<CLASSES>& mu, const CovMatrix& varInverse, double logVarDet,
                         double logPrior) {
	observation inverseMu = varInverse * mu;
	return inverseMu.dot(obs) - mu.dot(inverseMu) / 2.0 + logPrior;
}

double discriminateCase3(const observation& obs, const Vec<CLASSES>& mu, const CovMatrix& varInverse, double logVarDet,
                         double logPrior) {
	return (obs.dot((-1 / 2.0 * varInverse) * obs) + (varInverse * mu).dot(obs)) - 1 / 2.0 * mu.dot(varInverse * mu) -
	       logVarDet / 2.0 + logPrior;
}
#pragma endregion

unsigned detectCase(const array<CovMatrix, CLASSES>& vars) {
	// Default case is 3, since it covers all other cases as well
	unsigned discriminantCase = 3;

	// If all the covariance matrices are (approximately) equal, then we're in case 2
	if (std::equal(vars.cbegin() + 1, vars.cend(), vars.cbegin(),
	               [](const CovMatrix& mat1, const CovMatrix& mat2) { return mat1.isApprox(mat2); })) {
		discriminantCase = 2;

		// If the first covariance matrix is a scalar matrix (scalar times the identity), then we're in case 1
		if (vars[0].isApprox(vars[0](0, 0) * CovMatrix::Identity())) { discriminantCase = 1; }
	}

	return discriminantCase;
}

void calcInversesAndDets(unsigned discriminantCase, const array<CovMatrix, CLASSES>& vars,
                         array<CovMatrix, CLASSES>& varInverses, array<double, CLASSES>& varDets) {
	switch (discriminantCase) {
		case 1: {
			// Same inverse for all classes, and it's a diagonal matrix, so inverse is easy to find.
			// No need for determinant to discriminate
			CovMatrix inverse  = vars[0].diagonal().asDiagonal().inverse();
			double determinant = 1;

			std::for_each(varInverses.begin(), varInverses.end(), [&inverse](CovMatrix& inv) { inv = inverse; });
			std::for_each(varDets.begin(), varDets.end(), [&determinant](double& det) { det = determinant; });
			break;
		}
		case 2: {
			// Same inverse for all classes, but this time it's a bit more difficult to find
			// inverse. Covariance matrices are symmetric positive definite, so we can still find
			// inverse quickly as long as we don't need determinant (which we don't unless we're plotting the pdf).
			CovMatrix inverse  = vars[0].llt().solve(CovMatrix::Identity());
			double determinant = 1;

			std::for_each(varInverses.begin(), varInverses.end(), [&inverse](CovMatrix& inv) { inv = inverse; });
			std::for_each(varDets.begin(), varDets.end(), [&determinant](double& det) { det = determinant; });
			break;
		}
		case 3:
			// Different inverse for each class. They're still symmetric positive-definite, but now
			// we also need determinant and LLT decomposition doesn't give us that. So we use LU
			// decomposition, which takes longer than LLT but gives us both determinant and
			// inverse.
			std::transform(vars.cbegin(), vars.cend(), varInverses.begin(), varDets.begin(),
			               [](const CovMatrix& var, CovMatrix& inverse) {
				               PartialPivLU<CovMatrix> varLU = var.lu();

				               inverse = varLU.inverse();

				               return varLU.determinant();
			               });
			break;
	}
}

unsigned classifySample(unsigned discriminantCase, const sample& samp, unsigned correctClass, sample& misclass,
                        observation& min, observation& max, const array<observation, CLASSES>& means,
                        const array<CovMatrix, CLASSES>& varInverses, const array<double, CLASSES>& logVarDets,
                        const array<double, CLASSES>& logPriors) {
	unsigned misclassCount = 0;
	auto discriminate      = discriminantFuncs[discriminantCase - 1];

#pragma omp declare reduction(min:observation : omp_out = omp_out.cwiseMin(omp_in)) initializer(omp_priv(omp_orig))
#pragma omp declare reduction(max:observation : omp_out = omp_out.cwiseMax(omp_in)) initializer(omp_priv(omp_orig))
#pragma omp declare reduction(append:sample                                                  \
                              : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) \
    initializer(omp_priv(omp_orig))
#pragma omp parallel for reduction(+ : misclassCount) reduction(min : min) reduction(max : max) reduction(append : misclass)
	for (unsigned j = 0; j < samp.size(); j++) {
		double maxDiscriminant = discriminate(samp[j], means[0], varInverses[0], logVarDets[0], logPriors[0]);
		double discriminant;

		for (unsigned k = 1; k < CLASSES; k++) {
			discriminant = discriminate(samp[j], means[k], varInverses[k], logVarDets[k], logPriors[k]);

			if (k == correctClass && discriminant < maxDiscriminant ||
			    k > correctClass && discriminant > maxDiscriminant) {
				misclassCount++;

				break;
			}

			maxDiscriminant = std::max(maxDiscriminant, discriminant);
		}

		min = min.cwiseMin(samp[j]);
		max = max.cwiseMax(samp[j]);
	}

	return misclassCount;
}

void printPlotFile(std::ofstream& plotFile, const sample& samp) {
	plotFile << "#        x           y\n" << std::fixed << std::setprecision(7);

	for (unsigned j = 0; j < samp.size(); j++) {
		for (unsigned k = 0; k < DIM; k++) { plotFile << std::setw(10) << samp[j][k] << "  "; }
		plotFile << '\n';
	}
}

void printParamsFile(std::ofstream& boundaryParamsFile, const observation& min, const observation& max,
                     const array<observation, CLASSES>& means, const array<CovMatrix, CLASSES>& varInverses,
                     const array<double, CLASSES>& logVarDets, const array<double, CLASSES>& logPriors) {
	array<double, (DIM * (DIM + 1)) / 2 + DIM + 1> boundaryCoeffs;
	// Corresponds to the difference in the matrices "W_i" in the book
	CovMatrix diffW = -1 / 2.0 * (varInverses[0] - varInverses[1]);
	// Corresponds to the difference in the vectors "w_i" in the book
	observation diffw = varInverses[0] * means[0] - varInverses[1] * means[1];

	// Terms of degree 2 first, in alphabetical order
	// e.g. x^2 + xy + y^2,
	// or   x^2 + xy + xz + y^2 + yz + z^2
	for (unsigned i = 0; i < DIM; i++) {
		// Diagonals are the coefficients on squared terms
		boundaryCoeffs[i * DIM] = diffW(i, i);
		// Off-Diagonals are the coefficients on non-squared terms, and since the matrix is symmetric and xy = yx,
		// they get doubled.
		for (unsigned j = 1; j < DIM - i; j++) { boundaryCoeffs[i * DIM + j] = 2 * diffW(i, j + i); }
	}

	// Then terms of degree 1, once again in alphabetical order
	for (unsigned i = 0; i < DIM; i++) { boundaryCoeffs[(DIM * (DIM + 1)) / 2 + i] = diffw[i]; }

	// Then finally the constant term
	boundaryCoeffs.back() =
	    (-1 / 2.0 * means[0].dot(varInverses[0] * means[0]) - 1 / 2.0 * logVarDets[0] + logPriors[0]) -
	    (-1 / 2.0 * means[1].dot(varInverses[1] * means[1]) - 1 / 2.0 * logVarDets[1] + logPriors[1]);

	// Print parameter headers. All coefficients are single letters in alphabetical order, starting with 'a'
	for (unsigned i = 0; i < boundaryCoeffs.size(); i++) {
		boundaryParamsFile << std::setw(10) << (char) ('a' + i) << "  ";
	}
	// Then misc. parameter headers
	boundaryParamsFile << std::setw(10) << "xmin"
	                   << "  " << std::setw(10) << "xmax"
	                   << "  " << std::setw(10) << "ymin"
	                   << "  " << std::setw(10) << "ymax" << '\n'
	                   << std::fixed << std::setprecision(7);

	// Print parameters, starting with coefficients calculated above
	for (double coeff : boundaryCoeffs) { boundaryParamsFile << std::setw(10) << coeff << "  "; }
	// Then misc. parameters
	boundaryParamsFile << std::setw(10) << min[0] << "  " << std::setw(10) << max[0] << "  " << std::setw(10) << min[1]
	                   << "  " << std::setw(10) << max[1];
}

bool verifyArguments(int argc, char** argv, Arguments& arg, int& err) {
	if (argc < 2 || (argc < 2 && strcmp(argv[1], "-h") && strcmp(argv[1], "--help"))) {
		std::cout << "Missing operand.\n\n";
		err = 1;
		printHelp();
		return false;
	}

	if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
		printHelp();
		return false;
	}

	// Required arguments
	if (!strcmp(argv[1], "A") || !strcmp(argv[1], "a")) {
		arg.set = DataSet::A;
	} else if (!strcmp(argv[1], "B") || !strcmp(argv[1], "b")) {
		arg.set = DataSet::B;
	} else {
		std::cout << "Data Set \"" << argv[1] << "\" not recognised.\n\n";
		printHelp();
		err = 1;
		return false;
	}

	using namespace std::string_literals;
	std::regex samplePlotSwitch("-ps([1-"s + std::to_string(CLASSES) + "])"s);
	std::cmatch match;

	// Optional Arguments
	for (unsigned i = 2; i < argc; i++) {
		if (!strcmp(argv[i], "-s")) {
			if (i + 1 >= argc) {
				std::cout << "Missing seed value.\n\n";
				err = 1;
				printHelp();
				return false;
			}

			char* end;
			arg.seed = strtol(argv[i + 1], &end, 10);
			if (end == argv[i + 1]) {
				std::cout << "\"" << argv[i + 1] << "\" could not be interpreted as an integer.\n";
				err = 2;
				return false;
			}

			i++;
		} else if (!strcmp(argv[i], "-ps")) {
			if (i + 1 >= argc) {
				std::cout << "Missing training seed value.\n\n";
				err = 1;
				printHelp();
				return false;
			}

			char* end;
			arg.trainingSeed = strtol(argv[i + 1], &end, 10);
			if (end == argv[i + 1]) {
				std::cout << "\"" << argv[i + 1] << "\" could not be interpreted as an integer.\n";
				err = 2;
				return false;
			}

			i++;
		} else if (std::regex_match(argv[i], match, samplePlotSwitch)) {
			char* end;
			unsigned classNum = strtol(match[1].str().c_str(), &end, 10) - 1;

			if (i + 1 >= argc) {
				std::cout << "Missing sample " << classNum << " plot file.\n\n";
				err = 1;
				printHelp();
				return false;
			}

			arg.plotFiles[classNum].open(argv[i + 1]);
			if (!arg.plotFiles[classNum]) {
				std::cout << "Could not open file \"" << argv[i + 1] << "\".\n";
				err = 2;
				return false;
			}

			i++;
		} else if (!strcmp(argv[i], "-pdb")) {
			if (i + 1 >= argc) {
				std::cout << "Missing decision boundary parameter file.\n\n";
				err = 1;
				printHelp();
				return false;
			}

			arg.boundaryParamsFile.open(argv[i + 1]);
			if (!arg.boundaryParamsFile) {
				std::cout << "Could not open file \"" << argv[i + 1] << "\".\n";
				err = 2;
				return false;
			}

			i++;
		} else if (!strcmp(argv[i], "-c")) {
			if (i + 1 >= argc) {
				std::cout << "Missing case number.\n\n";
				err = 1;
				printHelp();
				return false;
			}

			char* end;
			arg.discriminantCase = strtol(argv[i + 1], &end, 10);
			if (end == argv[i + 1]) {
				std::cout << "\"" << argv[i + 1] << "\" could not be interpreted as an integer.\n";
				err = 2;
				return false;
			} else if (arg.discriminantCase < 1 || arg.discriminantCase > 3) {
				std::cout << "Discriminant case number must be between 1 and 3 (inclusive).\n";
				err = 2;
				return false;
			}

			i++;
		} else if (!strcmp(argv[i], "-p")) {
			if (i + 1 >= argc) {
				std::cout << "Missing sample percentage.\n\n";
				err = 1;
				printHelp();
				return false;
			}

			char* end;
			arg.samplePercent = strtod(argv[i + 1], &end) / 100;
			if (end == argv[i + 1]) {
				std::cout << "\"" << argv[i + 1] << "\" could not be interpreted as a floating-point number.\n";
				err = 2;
				return false;
			} else if (arg.samplePercent <= 0 || arg.samplePercent > 1) {
				std::cout << "Sample percentage must in (0,100].\n";
				err = 2;
				return false;
			}

			i++;
		} else if (!strcmp(argv[i], "-t")) {
			if (i + 1 >= argc) {
				std::cout << "Missing tabular data file.\n\n";
				err = 1;
				printHelp();
				return false;
			}

			arg.tabularDataFile.open(argv[i + 1], std::ios_base::in | std::ios_base::app);
			if (!arg.tabularDataFile) {
				std::cout << "Could not open file \"" << argv[i + 1] << "\".\n";
				err = 2;
				return false;
			}

			i++;
		} else if (!strcmp(argv[i], "-d")) {
			if (i + 1 >= argc) {
				std::cout << "Missing data input file.\n\n";
				err = 1;
				printHelp();
				return false;
			}

			arg.dataInputFile.open(argv[i + 1]);
			if (!arg.dataInputFile) {
				std::cout << "Could not open file \"" << argv[i + 1] << "\".\n";
				err = 2;
				return false;
			}

			i++;
		} else {
			std::cout << "Unrecognised argument \"" << argv[i] << "\".\n";
			printHelp();
			err = 1;
			return false;
		}
	}
	return true;
}

void printHelp() {
	Arguments arg;
	std::cout << "Usage: classify <data set> [options]                                         (1)\n"
	          << "   or: classify -h                                                           (2)\n\n"
	          << "(1) Run a Bayes classifier on a specific data set.\n"
	          << "    Data sets available are 'A' and 'B'.\n"
	          << "(2) Print this help menu.\n\n"
	          << "OPTIONS\n"
	          << "  -s   <seed>  Set the seed used to generate samples.\n"
	          << "               Defaults to " << arg.seed << ".\n"
	          << "  -psN <file>  Print all observations from sample N to a file.\n"
	          << "               N can be 1 to " << CLASSES << ".\n"
	          << "  -pmN <file>  Print all misclassified observations from sample N to a file.\n"
	          << "               N can be 1 to " << CLASSES << ".\n"
	          << "  -pdb <file>  Print the parameters of the decision boundary, along with\n"
	          << "               other miscellaneous parameters needed for plotting.\n"
	          << "  -pcb <file>  Print the parameters of the CORRECT (knowing the original\n"
	          << "               parameters) decision boundary.\n"
	          << "  -c   <case>  Override which discriminant case is to be used.\n"
	          << "               <case> can be 1-3. Higher numbers are more computationally\n"
	          << "               expensive, but are correct more of the time.\n"
	          << "               By default, the case will be chosen automatically.\n"
	          << "  -p   <num>   The percent of the samples generated to consider when\n"
	          << "               generating the distribution parameters.\n"
	          << "               Defaults to " << (arg.samplePercent * 100) << "%.\n"
	          << "  -ps  <seed>  Set the seed to use when choosing samples to consider for\n"
	          << "               generating the distribution parameters.\n"
	          << "               Defaults to " << arg.trainingSeed << ".\n"
	          << "  -t   <file>  Print tabulated data about the performance of the classifier\n"
	          << "               to a file to be used in LaTeX. Only prints one row of the\n"
	          << "               table, appended to the end of the file for use in comparing\n"
	          << "               other parameters.\n"
	          << "  -d   <file>  Instead of generating new samples, load them from this file\n"
	          << "               The file is expected to be a space-separated values file,\n"
	          << "               where the first n values are the features of a particular\n"
	          << "               observation, and the last value is the integer class label,\n"
	          << "               which is 0-indexed.\n";
}