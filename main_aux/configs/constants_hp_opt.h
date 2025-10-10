#pragma once

#include "main_aux/configs/constants_algorithms.h"

#include <unordered_set>


// ---------------------------------------------------------------------------------------------------------------------
// For hyperparam optimisation: set of integer parameters
// ---------------------------------------------------------------------------------------------------------------------

static const std::unordered_set<std::string> HPOPT_INTEGER_PARAM_IDS =
{
    ALG_PARAM_ID_UCT_BUDGET,
};

// ---------------------------------------------------------------------------------------------------------------------
// For hyperparam optimisation: set of parameters to search over a log scaling
// ---------------------------------------------------------------------------------------------------------------------

static const std::unordered_set<std::string> HPOPT_LOG_SCALE_PARAM_IDS =
{
    ALG_PARAM_ID_BIAS,
    ALG_PARAM_ID_UCT_BUDGET,
    ALG_PARAM_ID_INIT_TEMP,
    ALG_PARAM_ID_TEMP_DECAY_RATE,
    ALG_PARAM_ID_INIT_ENTROPY_COEFF,
    ALG_PARAM_ID_ENTROPY_COEFF_DECAY_RATE,
    ALG_PARAM_ID_EPSILON,
};