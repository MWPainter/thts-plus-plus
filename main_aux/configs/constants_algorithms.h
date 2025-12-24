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

static const std::string ALG_ID_UCT = "uct";
static const std::string ALG_ID_MAX_UCT = "maxuct";
static const std::string ALG_ID_HMCTS = "hmcts";
static const std::string ALG_ID_MENTS = "ments";
static const std::string ALG_ID_RENTS = "rents";
static const std::string ALG_ID_TENTS = "tents";
static const std::string ALG_ID_BTS = "bts";
static const std::string ALG_ID_DENTS = "dents";

// ---------------------------------------------------------------------------------------------------------------------
// Constants to identify parameters used by algoirithms
// ---------------------------------------------------------------------------------------------------------------------

static const std::string ALG_PARAM_ID_BIAS = "bias";                                        // bias param (uct algorithms)
static const std::string ALG_PARAM_ID_UCT_BUDGET = "uct_budget";                            // uct budget (for hmcts)
static const std::string ALG_PARAM_ID_INIT_TEMP = "temp";                                   // initial temp (boltzmann algorithms)
static const std::string ALG_PARAM_ID_TEMP_DECAY_RATE= "temp_decay_rate";                   // param controling temp decay (boltzmann with decay algorithms)
static const std::string ALG_PARAM_ID_INIT_ENTROPY_COEFF = "entropy_coeff";                 // initial entropy coeff (dents)
static const std::string ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT = "entropy_zero_at";            // number of trials after which entropy_coeff linearly decays to zero (dents)
static const std::string ALG_PARAM_ID_EPSILON = "epsilon";                                  // exploration param (boltzmann algorithms)
static const std::string ALG_PARAM_ID_DEFAULT_Q_VALUE = "default_q_value";                  // default value of Q(s,a) for unseen state action pairs (boltzmann algorithms)

// ---------------------------------------------------------------------------------------------------------------------
// A map specifying the relevant parameters for each algorithm (all the params that should be specified)
// ---------------------------------------------------------------------------------------------------------------------

static const std::unordered_map<std::string,std::vector<std::string>> ALG_ID_TO_ALG_PARAM_IDS =
{
    {ALG_ID_UCT,
        {
            ALG_PARAM_ID_BIAS,
        },
    },
    {ALG_ID_MAX_UCT,
        {
            ALG_PARAM_ID_BIAS,
        },
    },
    {ALG_ID_HMCTS,
        {
            ALG_PARAM_ID_BIAS,
            ALG_PARAM_ID_UCT_BUDGET,
        },
    },
    {ALG_ID_MENTS,
        {
            ALG_PARAM_ID_INIT_TEMP,
            // ALG_PARAM_ID_TEMP_DECAY_RATE,
            ALG_PARAM_ID_EPSILON,
            ALG_PARAM_ID_DEFAULT_Q_VALUE,
        },
    },
    {ALG_ID_RENTS,
        {
            ALG_PARAM_ID_INIT_TEMP,
            // ALG_PARAM_ID_TEMP_DECAY_RATE,
            ALG_PARAM_ID_EPSILON,
            ALG_PARAM_ID_DEFAULT_Q_VALUE,
        },
    },
    {ALG_ID_TENTS,
        {
            ALG_PARAM_ID_INIT_TEMP,
            // ALG_PARAM_ID_TEMP_DECAY_RATE,
            ALG_PARAM_ID_EPSILON,
            ALG_PARAM_ID_DEFAULT_Q_VALUE,
        },
    },
    {ALG_ID_BTS,
        {
            ALG_PARAM_ID_INIT_TEMP,
            ALG_PARAM_ID_TEMP_DECAY_RATE,
            ALG_PARAM_ID_EPSILON,
            ALG_PARAM_ID_DEFAULT_Q_VALUE,
        },
    },
    {ALG_ID_DENTS,
        {
            ALG_PARAM_ID_INIT_TEMP,
            ALG_PARAM_ID_TEMP_DECAY_RATE,
            ALG_PARAM_ID_INIT_ENTROPY_COEFF,
            ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,
            ALG_PARAM_ID_EPSILON,
            ALG_PARAM_ID_DEFAULT_Q_VALUE,
        },
    },
};