#pragma once

#include "main_mo/configs/constants_algorithms.h"

#include <unordered_set>

/**
 * This file defines HPOPT_PARAM_ID constants
 * - HPOPT_PARAM_ID = id's for running hyperparameter optimization
 */

// ---------------------------------------------------------------------------------------------------------------------
// Tags to indicate what a config dictionary contains to indicate hpopt/experiment level params in configs
// ---------------------------------------------------------------------------------------------------------------------

static const std::string HPOPT_PARAMS_ID_TAG = "hptopt_params";                 // Special value to indicate dictionary is specifying xpr level params

// ---------------------------------------------------------------------------------------------------------------------
// Constants used in the hyperparameter optimisation, but might vary on a per environment basis
// ---------------------------------------------------------------------------------------------------------------------

static const std::string HPOPT_PARAM_ID_MIN_REPEATS = "min_repeats";                                // minimum number of times to run alg with params before returning estimate for the sampled params
static const std::string HPOPT_PARAM_ID_ESTIMATE_CONFIDENCE_THRESHOLD = "conf_thresh";              // minimum threshold for std of evals before returning estimate for sampled params (keep running algorithm until empirical std of evals is below threshold)
static const std::string HPOPT_PARAM_ID_BAYESOPT_TOTAL_SAMPLES = "bayesopt_total_samples";          // total number of samples to use in bayesopt
static const std::string HPOPT_PARAM_ID_BAYESOPT_INIT_RAND_SAMPLES = "bayesopt_init_rand_samples";  // how many samples to sample uniformly randomly at start of bayesopt
static const std::string HPOPT_PARAM_ID_BAYESOPT_RELEARN_FREQ = "bayesopt_relearn_freq";            // how often bayesopt will internally update or "relearn" parameters used in bayesian optimisation
static const std::string HPOPT_PARAM_ID_BAYESOPT_USE_GPML = "bayesopt_use_gp_ml";                  // if should use Gaussian Process surrogate model with maximum likelihood estimation

// ---------------------------------------------------------------------------------------------------------------------
// For hyperparam optimisation: set of parameters to search over a log scaling
// ---------------------------------------------------------------------------------------------------------------------

static const std::unordered_set<std::string> HPOPT_LOG_SCALE_ALG_PARAM_IDS =
{
    ALG_PARAM_ID_BIAS,
    ALG_PARAM_ID_INIT_TEMP,
    ALG_PARAM_ID_TEMP_DECAY_RATE,
    ALG_PARAM_ID_INIT_ENTROPY_COEFF,
    ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,
    ALG_PARAM_ID_EPSILON,
};

// ---------------------------------------------------------------------------------------------------------------------
// For hyperparam optimisation: set of parameters that we need an integer value
// ---------------------------------------------------------------------------------------------------------------------

static const std::unordered_set<std::string> HPOPT_INT_ALG_PARAM_IDS =
{
    ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,
    ALG_PARAM_ID_MIN_LOG2_N,
};
