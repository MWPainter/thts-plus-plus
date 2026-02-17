#pragma once

#include <string>
#include <unordered_map>
#include <vector>


/**
 * This file defines ALG_ID constants and ALG_PARAM_ID constants
 * - ALG_ID = id's for algorithms
 * - ALG_PARAM_ID = id's for algorithm parameters
 */


// ---------------------------------------------------------------------------------------------------------------------
// Constants to identify different algorithms
// ---------------------------------------------------------------------------------------------------------------------

static const std::string ALG_ID_CHVI = "chvi";

static const std::string ALG_ID_CZT = "czt";
static const std::string ALG_ID_CZT_DOUBLING = "czt_doubling";
static const std::string ALG_ID_CH_UCT = "ch_uct";
static const std::string ALG_ID_CH_CZT = "ch_czt";
static const std::string ALG_ID_CH_CZT_DOUBLING = "ch_czt_doubling";
static const std::string ALG_ID_CH_BTS = "ch_bts";
static const std::string ALG_ID_CH_DENTS = "ch_dents";
static const std::string ALG_ID_CH_HVUCT = "ch_hvuct";
static const std::string ALG_ID_CH_PARETO = "ch_pareto";
static const std::string ALG_ID_CH_CHEBY = "ch_cheby";
static const std::string ALG_ID_CH_STANDARD_CHEBY = "ch_standard_cheby";

// ---------------------------------------------------------------------------------------------------------------------
// Constants to identify parameters used by algoirithms
// ---------------------------------------------------------------------------------------------------------------------

static const std::string ALG_PARAM_ID_VECTOR_VISIT_COUNTS = "vector_visit_counts";                  // whether to use vector visit counts
static const std::string ALG_PARAM_ID_BIAS = "bias";                                                // bias param (uct algorithms)
static const std::string ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH = "czt_ball_split_visit_thresh";  // minimum number of visits at a ball in czt algorithms before alowed to split 
static const std::string ALG_PARAM_ID_MIN_LOG2_N = "min_log2_n";                                    // minimum value of N (total num trials estimate) for doubling czt
static const std::string ALG_PARAM_ID_INIT_TEMP = "temp";                                           // initial temp (boltzmann algorithms)
static const std::string ALG_PARAM_ID_TEMP_DECAY_RATE= "temp_decay_rate";                           // param controling temp decay (boltzmann with decay algorithms)
static const std::string ALG_PARAM_ID_INIT_ENTROPY_COEFF = "entropy_coeff";                         // initial entropy coeff (dents)
static const std::string ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT = "entropy_zero_at";                    // number of trials after which entropy_coeff linearly decays to zero (dents)
static const std::string ALG_PARAM_ID_EPSILON = "epsilon";                                          // exploration param (boltzmann algorithms)
static const std::string ALG_PARAM_ID_DEFAULT_Q_VALUE = "default_q_value";                          // default value of Q(s,a) for unseen state action pairs (boltzmann algorithms)

// ---------------------------------------------------------------------------------------------------------------------
// A map specifying the relevant parameters for each algorithm (all the params that should be specified)
// ---------------------------------------------------------------------------------------------------------------------

static const std::unordered_map<std::string,std::vector<std::string>> ALG_ID_TO_ALG_PARAM_IDS =
{
    {ALG_ID_CHVI,
        {
        },
    },
    {ALG_ID_CZT,
        {
            ALG_PARAM_ID_BIAS,
            ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,
        },
    },
    {ALG_ID_CZT_DOUBLING,
        {
            ALG_PARAM_ID_BIAS,
            ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,
            ALG_PARAM_ID_MIN_LOG2_N,
        },
    },
    {ALG_ID_CH_UCT,
        {
            ALG_PARAM_ID_BIAS,
        },
    },
    {ALG_ID_CH_CZT,
        {
            ALG_PARAM_ID_BIAS,
            ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,
        },
    },
    {ALG_ID_CH_CZT_DOUBLING,
        {
            ALG_PARAM_ID_BIAS,
            ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,
            ALG_PARAM_ID_MIN_LOG2_N,
        },
    },
    {ALG_ID_CH_BTS,
        {
            ALG_PARAM_ID_INIT_TEMP,
            ALG_PARAM_ID_TEMP_DECAY_RATE,
            ALG_PARAM_ID_EPSILON,
            ALG_PARAM_ID_DEFAULT_Q_VALUE,
        },
    },
    {ALG_ID_CH_DENTS,
        {
            ALG_PARAM_ID_INIT_TEMP,
            ALG_PARAM_ID_TEMP_DECAY_RATE,
            ALG_PARAM_ID_INIT_ENTROPY_COEFF,
            ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,
            ALG_PARAM_ID_EPSILON,
            ALG_PARAM_ID_DEFAULT_Q_VALUE,
        },
    },
    {ALG_ID_CH_HVUCT,
        {
            ALG_PARAM_ID_BIAS,
        },
    },
    {ALG_ID_CH_PARETO,
        {
            ALG_PARAM_ID_BIAS,
        },
    },
    {ALG_ID_CH_CHEBY,
        {
            ALG_PARAM_ID_BIAS,
        },
    },
    {ALG_ID_CH_STANDARD_CHEBY,
        {
            ALG_PARAM_ID_BIAS,
        },
    },
};