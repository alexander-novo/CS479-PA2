#pragma once
#include <omp.h>

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <iomanip>
#include <random>

using Eigen::aligned_allocator;
using Eigen::Matrix;
using Eigen::SelfAdjointEigenSolver;

template <unsigned N>
using Vec = Matrix<double, N, 1u>;

// #pragma omp declare reduction(+: Matrix<double, N, M> : omp_out += omp_in)
// initializer(omp_priv(omp_orig))

/**
 * @brief Generate a multivariate-normally-distributed random sample with the given
 *        parameters. Special case with the covariance matrix diagonal (i.e. the
 *        variables are uncorrelated). Multi-threaded.
 *
 * @param      mu      The mean of the distribution
 * @param      sigma   The standard deviation of the distribution
 * @param[out] sample  Where the sample outputs are stored. The sample size
 *                     is the size of the vector.
 * @param      seed    What to seed the RNG with. Defaults to 1.
 */
template <unsigned N>
void genGaussianSample(const Vec<N>& mu, const Matrix<double, N, N>& Sigma,
                       std::vector<Vec<N>, aligned_allocator<Vec<N>>>& sample, unsigned seed = 1) {
	SelfAdjointEigenSolver<Matrix<double, N, N>> solver(Sigma);

	Matrix<double, N, N> inverseWhitening = solver.operatorSqrt() * solver.eigenvectors();
#pragma omp parallel
	{
		// Seed must be partially based on thread ID, otherwise each thread will gen
		// same numbers.
		std::mt19937_64 engine(seed + omp_get_thread_num());
		std::normal_distribution<double> dist;

		// Initialise distributions with given information
		// for (unsigned i = 0; i < N; i++) { dists.emplace_back(mu[i], sigma[i]); }

#pragma omp for
		// Generate random observations. Since the variables are uncorrelated
		// from a multivariate normal distribution, they are also independent.
		for (unsigned j = 0; j < sample.size(); j++) {
			for (unsigned i = 0; i < N; i++) { sample[j][i] = dist(engine); }
			sample[j] = inverseWhitening * sample[j] + mu;
		}
	}
}

/**
 * @brief The probability density function (pdf) for the multivariate gaussian (normal) distribution.
 *
 * @tparam N            The dimension of the distribution.
 * @param x             Where to evaluate the density function at.
 * @param mu            The mean of the distribution.
 * @param SigmaInverse  The inverse of the covariance matrix of the distribution.
 * @param SigmaDetRoot  The square root of the determinant of the distribution.
 * @return double       The value of the pdf evaluated at x.
 */
template <unsigned N>
double gaussianDensity(const Vec<N>& x, const Vec<N>& mu, const Matrix<double, N, N>& SigmaInverse,
                       double SigmaDetRoot) {
	Vec<N> muDiff = x - mu;
	return 1 / pow(2 * M_PI, N / 2.0) / SigmaDetRoot * exp(-1 / 2.0 * muDiff.dot(SigmaInverse * muDiff));
}