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
	getSamples(arg.set, samples, arg.seed);

	array<observation, CLASSES> means = getSampleMeans(samples);
	array<CovMatrix, CLASSES> vars    = getSampleVars(samples, means);

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

	calcInversesAndDets(arg.discriminantCase, vars, varInverses, varDets, !arg.pdfPlotFile.fail());

	// Calculate the logs of the determinants
	std::transform(varDets.cbegin(), varDets.cend(), logVarDets.begin(), [](const double& det) { return log(det); });

	unsigned overallMisclass = 0;

	std::cout << "Misclassification rates:\n";
	for (unsigned i = 0; i < CLASSES; i++) {
		sample& misclass = misclassifications[i];

		unsigned misclassCount =
		    classifySample(arg.discriminantCase, samples[i], i, misclassifications[i], min, max, means, varInverses,
		                   logVarDets, logPriors, !arg.misclassPlotFiles[i].fail());

		std::cout << misclassCount / (double) sizes[i] << "\n";

		overallMisclass += misclassCount;

		if (arg.plotFiles[i]) { printPlotFile(arg.plotFiles[i], samples[i]); }

		if (arg.misclassPlotFiles[i]) { printPlotFile(arg.misclassPlotFiles[i], misclassifications[i]); }
	}

	std::cout << "\nOverall misclassification rate:\n" << overallMisclass / totalSize << "\n\n";

	double bhattacharyyaBound = errorBoundFunc(.5, means, vars, varDets, priors);
	double chernoffBound;

	double minBound = 0, maxBound = 1, prospectiveBound = 0.5;
	double minBoundVal         = errorBoundFuncDiff(minBound, means, vars, varDets, priors);
	double maxBoundVal         = errorBoundFuncDiff(maxBound, means, vars, varDets, priors);
	double prospectiveBoundVal = errorBoundFuncDiff(prospectiveBound, means, vars, varDets, priors);
	unsigned iters             = 0;

	if (prospectiveBoundVal < 0) {
		minBoundVal = prospectiveBoundVal;
		minBound    = prospectiveBound;
	} else if (prospectiveBoundVal > 0) {
		maxBoundVal = prospectiveBoundVal;
		maxBound    = prospectiveBound;
	}

	double guess[]      = {0.5, (minBound + maxBound) / 2.};
	double oldGuessVal  = prospectiveBoundVal;
	prospectiveBoundVal = errorBoundFuncDiff(guess[1], means, vars, varDets, priors);

	while (abs(prospectiveBoundVal) > ERROR_BOUND_MIN_EPSILON && iters < ERROR_BOUND_MAX_ITERS) {
		// Update bounds
		if (prospectiveBoundVal < 0) {
			minBoundVal = prospectiveBoundVal;
			minBound    = guess[1];
		} else if (prospectiveBoundVal > 0) {
			maxBoundVal = prospectiveBoundVal;
			maxBound    = guess[1];
		}

		// Find the next guess from secant method
		double nextGuess = guess[1] - prospectiveBoundVal * (guess[1] - guess[0]) / (prospectiveBoundVal - oldGuessVal);

		// If the next guess from the secant method is outside our bounds, it is attempting to diverge.
		// Use bisection method instead.
		if (nextGuess < minBound || nextGuess > maxBound) { nextGuess = (minBound + maxBound) / 2.0; }

		// Advance the iteration
		guess[0] = guess[1];
		guess[1] = nextGuess;

		oldGuessVal         = prospectiveBoundVal;
		prospectiveBoundVal = errorBoundFuncDiff(guess[1], means, vars, varDets, priors);

		iters++;
	}

	chernoffBound = errorBoundFunc(guess[1], means, vars, varDets, priors);

	std::cout << "Bhattacharyya Bound:\n"
	          << bhattacharyyaBound << "\n\n"
	          << "Chernoff Bound:\n"
	          << chernoffBound << " @ beta = " << guess[1] << "\n\n";

	if (arg.errorBoundFile) { printErrorBoundFile(arg.errorBoundFile, means, vars, varDets, priors); }

	double pdfMax = 0;
	if (arg.pdfPlotFile) {
		pdfMax = printPdfPlotFile(arg.pdfPlotFile, min, max, means, varInverses, varDets, priors);

		// Round it so that the z tick values are not so ugly
		pdfMax = round(pdfMax * 100) / 100;
	}

	if (arg.boundaryParamsFile) {
		printParamsFile(arg.boundaryParamsFile, min, max, means, varInverses, logVarDets, logPriors, pdfMax,
		                bhattacharyyaBound, chernoffBound, guess[1]);
	}

	return 0;
}

array<observation, CLASSES> getSampleMeans(const array<sample, CLASSES>& samples) {
	array<observation, CLASSES> means;

	for (unsigned i = 0; i < CLASSES; i++) { means[i] = sampleMean(samples[i]); }

	return means;
}

array<CovMatrix, CLASSES> getSampleVars(const array<sample, CLASSES>& samples,
                                        const array<observation, CLASSES>& sampleMeans) {
	array<CovMatrix, CLASSES> vars;

	for (unsigned i = 0; i < CLASSES; i++) { vars[i] = sampleVariance(samples[i], sampleMeans[i]); }

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
                         array<CovMatrix, CLASSES>& varInverses, array<double, CLASSES>& varDets, bool alwaysCalcDets) {
	switch (discriminantCase) {
		case 1: {
			// Same inverse for all classes, and it's a diagonal matrix, so inverse is easy to find.
			// No need for determinant to discriminate, but we need it if we're plotting the pdf.
			CovMatrix inverse = vars[0].diagonal().asDiagonal().inverse();

			// The determinant of a diagonal matrix is the product of the entries on the main diagonal.
			// But since this is a scalar matrix, the entries are all the same so it is the power of that scalar.
			double determinant = alwaysCalcDets ? determinant = pow(vars[0](0, 0), DIM) : 1;

			std::for_each(varInverses.begin(), varInverses.end(), [&inverse](CovMatrix& inv) { inv = inverse; });
			std::for_each(varDets.begin(), varDets.end(), [&determinant](double& det) { det = determinant; });
			break;
		}
		case 2: {
			// Same inverse for all classes, but this time it's a bit more difficult to find
			// inverse. Covariance matrices are symmetric positive definite, so we can still find
			// inverse quickly as long as we don't need determinant (which we don't unless we're plotting the pdf).
			CovMatrix inverse;
			double determinant;
			if (!alwaysCalcDets) {
				inverse     = vars[0].llt().solve(CovMatrix::Identity());
				determinant = 1;
			} else {
				// See below why we don't use LLT decomposition in this case
				PartialPivLU<CovMatrix> varLU = vars[0].lu();
				inverse                       = varLU.inverse();
				determinant                   = varLU.determinant();
			}

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
                        const array<double, CLASSES>& logPriors, bool plotMisclassifications) {
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

				if (plotMisclassifications) { misclass.push_back(samp[j]); }

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

void printErrorBoundFile(std::ofstream& plotFile, const array<observation, 2>& means, const array<CovMatrix, 2>& vars,
                         const array<double, 2>& varDets, const array<double, 2>& priors) {
	plotFile << "#        x           y          y'\n" << std::fixed << std::setprecision(7);

	for (unsigned i = 0; i < ERROR_BOUND_SAMPLES; i++) {
		double x  = i / ((double) ERROR_BOUND_SAMPLES - 1);
		double y  = errorBoundFunc(x, means, vars, varDets, priors);
		double dy = errorBoundFuncDiff(x, means, vars, varDets, priors);
		plotFile << std::setw(10) << x << "  " << std::setw(10) << y << "  " << std::setw(10) << dy << '\n';
	}
}

double printPdfPlotFile(std::ofstream& pdfPlotFile, const observation& min, const observation& max,
                        const array<observation, CLASSES>& means, const array<CovMatrix, CLASSES>& varInverses,
                        const array<double, CLASSES>& varDets, const array<double, CLASSES>& priors) {
	double pdfMax = 0;
	array<double, CLASSES> varDetRoots;

	std::transform(varDets.cbegin(), varDets.cend(), varDetRoots.begin(), [](double det) { return sqrt(det); });

	pdfPlotFile << "#        x           y           z       class\n" << std::fixed << std::setprecision(7);
#pragma omp parallel for ordered collapse(2) reduction(max : pdfMax)
	for (unsigned x = 0; x < PDF_SAMPLES; x++) {
		for (unsigned y = 0; y < PDF_SAMPLES; y++) {
			double xmod = min.x() + x * (max.x() - min.x()) / PDF_SAMPLES;
			double ymod = min.y() + y * (max.y() - min.y()) / PDF_SAMPLES;

			observation vecX = {xmod, ymod};

			array<double, CLASSES> densities;

			for (unsigned i = 0; i < CLASSES; i++) {
				densities[i] = gaussianDensity<DIM>(vecX, means[i], varInverses[i], varDetRoots[i]) * priors[i];
			}

			double jointDensity = std::accumulate(densities.cbegin(), densities.cend(), 0.);
			unsigned correctClass =
			    std::distance(densities.cbegin(), std::max_element(densities.cbegin(), densities.cend())) + 1;

			pdfMax = std::max(pdfMax, jointDensity);

#pragma omp ordered
			{
				pdfPlotFile << std::setw(10) << xmod << "  " << std::setw(10) << ymod << "  " << std::setw(10)
				            << jointDensity << "  " << std::setw(10) << correctClass << '\n';

				// gnuplot requires an extra blank line between rows on surface plots
				if (y == PDF_SAMPLES - 1) { pdfPlotFile << '\n'; }
			}
		}
	}

	return pdfMax;
}

void printParamsFile(std::ofstream& boundaryParamsFile, const observation& min, const observation& max,
                     const array<observation, CLASSES>& means, const array<CovMatrix, CLASSES>& varInverses,
                     const array<double, CLASSES>& logVarDets, const array<double, CLASSES>& logPriors, double pdfMax,
                     double bhattacharyyaBound, double chernoffBound, double betaStar) {
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
	                   << "  " << std::setw(10) << "ymax"
	                   << "  " << std::setw(10) << "zmax"
	                   << "  " << std::setw(10) << "boundB"
	                   << "  " << std::setw(10) << "boundC"
	                   << "  " << std::setw(10) << "betaStar" << '\n'
	                   << std::fixed << std::setprecision(7);

	// Print parameters, starting with coefficients calculated above
	for (double coeff : boundaryCoeffs) { boundaryParamsFile << std::setw(10) << coeff << "  "; }
	// Then misc. parameters
	boundaryParamsFile << std::setw(10) << min[0] << "  " << std::setw(10) << max[0] << "  " << std::setw(10) << min[1]
	                   << "  " << std::setw(10) << max[1] << "  " << std::setw(10) << pdfMax << "  " << std::setw(10)
	                   << bhattacharyyaBound << "  " << std::setw(10) << chernoffBound << "  " << std::setw(10)
	                   << betaStar;
}

double errorBoundFunc(double beta, const array<observation, 2>& means, const array<CovMatrix, 2>& vars,
                      const array<double, 2>& dets, const array<double, 2>& priors) {
	observation diffMu                    = means[0] - means[1];
	CovMatrix weightedVar                 = (1 - beta) * vars[0] + beta * vars[1];
	PartialPivLU<CovMatrix> weightedVarLU = weightedVar.lu();

	return pow(priors[0], beta) * pow(priors[1], 1 - beta) *
	       exp(-((beta * (1 - beta)) / 2. * diffMu.dot(weightedVarLU.inverse() * diffMu) +
	             1 / 2. * log(weightedVarLU.determinant() / pow(dets[0], 1 - beta) / pow(dets[1], beta))));
}

double errorBoundFuncDiff(double beta, const array<observation, 2>& means, const array<CovMatrix, 2>& vars,
                          const array<double, 2>& dets, const array<double, 2>& priors) {
	observation diffMu                    = means[0] - means[1];
	CovMatrix weightedVar                 = (1 - beta) * vars[0] + beta * vars[1];
	PartialPivLU<CovMatrix> weightedVarLU = weightedVar.lu();
	CovMatrix weightedVarInverse          = weightedVarLU.inverse();
	double weightedVarDistance            = diffMu.dot(weightedVarLU.inverse() * diffMu);
	CovMatrix weightedVarInnerDiff        = -vars[0] + vars[1];
	double logInner                       = weightedVarLU.determinant() / pow(dets[0], 1 - beta) / pow(dets[1], beta);

	// Triple product rule original functions (f*g*h)' = f'gh + fg'h + fgh'
	double f = pow(priors[0], beta), g = pow(priors[1], 1 - beta),
	       h  = exp(-((beta * (1 - beta)) / 2. * weightedVarDistance + 1 / 2. * log(logInner)));
	double df = f * log(priors[0]), dg = -g * log(priors[1]);

	// Triple quotient rule original functions (f / g / h)' = (f'gh-fg'h - fgh') / g^2 / h^2
	// For calculating h' above
	double f2 = weightedVarLU.determinant();
	double g2 = pow(dets[0], 1 - beta);
	double h2 = pow(dets[1], beta);

	// Jacobi's formula
	double df2 = weightedVarLU.determinant() * (weightedVarInverse * weightedVarInnerDiff).trace(),
	       dg2 = -g2 * log(dets[0]), dh2 = h2 * log(dets[1]);

	double tripleQuotientDiff = (df2 * g2 * h2 - f2 * dg2 * h2 - f2 * g2 * dh2) / (g2 * g2 * h2 * h2);

	// The first term of k is also a triple product, but the first 2 derivatives are simple, so it has jsut been written
	// out without intermediate variables. Note that (A^{-1})' = -A^{-1}A'A^{-1} when dealing with matrices.
	double dk = (1 - beta) / 2. * weightedVarDistance - beta / 2. * weightedVarDistance +
	            (beta * (1 - beta)) / 2. *
	                diffMu.dot((-weightedVarInverse * weightedVarInnerDiff * weightedVarInverse) * diffMu) +
	            1 / 2. * (tripleQuotientDiff) / (logInner);

	double dh = h * (-dk);

	return df * g * h + f * dg * h + f * g * dh;
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
	std::regex misclassPlotSwitch("-pm([1-"s + std::to_string(CLASSES) + "])"s);
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
		} else if (std::regex_match(argv[i], match, misclassPlotSwitch)) {
			char* end;
			unsigned classNum = strtol(match[1].str().c_str(), &end, 10) - 1;

			if (i + 1 >= argc) {
				std::cout << "Missing misclassifications of sample " << classNum << " plot file.\n\n";
				err = 1;
				printHelp();
				return false;
			}

			arg.misclassPlotFiles[classNum].open(argv[i + 1]);
			if (!arg.misclassPlotFiles[classNum]) {
				std::cout << "Could not open file \"" << argv[i + 1] << "\".\n";
				err = 2;
				return false;
			}

			i++;
		} else if (!strcmp(argv[i], "-pdf")) {
			if (i + 1 >= argc) {
				std::cout << "Missing probability density function file.\n\n";
				err = 1;
				printHelp();
				return false;
			}

			arg.pdfPlotFile.open(argv[i + 1]);
			if (!arg.pdfPlotFile) {
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
		} else if (!strcmp(argv[i], "-peb")) {
			if (i + 1 >= argc) {
				std::cout << "Missing error bound plot file.\n\n";
				err = 1;
				printHelp();
				return false;
			}

			arg.errorBoundFile.open(argv[i + 1]);
			if (!arg.errorBoundFile) {
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
	std::cout << "Usage: classify-bayes <data set> [options]                            (1)\n"
	          << "   or: classify-bayes -h                                              (2)\n\n"
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
	          << "  -pdf <file>  Print a graph of the probability density function to a file.\n"
	          << "               There will be an extra column which shows which class is more\n"
	          << "               likely at that point. Will also allow correct calculation of\n"
	          << "               zmax in the -pdb file.\n"
	          << "  -peb <file>  Print a graph of the error bound function and its derivative\n"
	          << "               to a file.\n"
	          << "  -pdb <file>  Print the parameters of the decision boundary, along with\n"
	          << "               other miscellaneous parameters needed for plotting.\n"
	          << "  -c   <case>  Override which discriminant case is to be used.\n"
	          << "               <case> can be 1-3. Higher numbers are more computationally\n"
	          << "               expensive, but are correct more of the time.\n"
	          << "               By default, the case will be chosen automatically.\n";
}