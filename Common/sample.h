#pragma once
#include "gauss.h"

#define CLASSES 2u
#define DIM 2u

using observation = Vec<DIM>;
using CovMatrix   = Matrix<double, DIM, DIM>;
using sample      = std::vector<observation, aligned_allocator<observation>>;
using std::array;

#pragma omp declare reduction(+ : observation : omp_out = omp_out + omp_in) initializer(omp_priv(omp_orig))
#pragma omp declare reduction(+ : CovMatrix : omp_out = omp_out + omp_in) initializer(omp_priv(omp_orig))

// The data sets from the assignment
enum DataSet { A, B };

/**
 * @brief Retreives samples from distributions as labeled by the assignment.
 *
 * @param set      What distributions to use.
 * @param samples
 * @param seed     Seed the RNG. Keep consistent for the same samples.
 */
void getSamples(DataSet set, array<sample, CLASSES>& samples, unsigned seed = 1);

/**
 * @brief Retrieve the means of a data set.
 *
 * @param set                                The data set to get the means of.
 * @return array<observation, CLASSES>  The means.
 */
array<observation, CLASSES> getMeans(DataSet set);

/**
 * @brief Retrieve the variances of a data set. The covariance matrix of all samples
 *        in all data sets are diagonal, so the variances are just the diagonal
 *        elements.
 *
 * @param set                              The data set to get the variances of.
 * @return array<CovMatrix, CLASSES>  The variances.
 */
array<CovMatrix, CLASSES> getVars(DataSet set);

/**
 * @brief Retrieve the number of observations for each sample in a data set.
 *
 * @param set                             The data set to get the sizes of.
 * @return array<unsigned, CLASSES>  The sizes.
 */
array<unsigned, CLASSES> getSizes(DataSet set);

/**
 * @brief Retrieve the name of the data set in string form
 *
 * @param set           The data set to get the name of
 * @return std::string  The name of the data set
 */
std::string dataSetName(DataSet set);

/**
 * @brief Calculate the sample mean from a sample.
 *
 * @tparam N       The number of features in the sample.
 * @param sample   The sample to calculate from.
 * @return Vec<N>  The sample mean.
 */
observation sampleMean(const sample& sample);

/**
 * @brief Calculate the sample variance from a sample. Uses Bessel's correction for
 *        unbiased-ness.
 *
 * @tparam N          The number of features in the sample.
 * @param sample      The sample to calculate from.
 * @param sampleMean  The sample mean.
 *                    See sampleMean(sample).
 * @return            The sample variance.
 */
CovMatrix sampleVariance(const sample& sample, const observation& sampleMean);