#pragma once

#include "main_aux/configs/constants_algorithms.h"

#include <unordered_set>
#include <utility>
#include <variant>

/**
 * This file defines HPOPT_PARAM_ID constants
 * - HPOPT_PARAM_ID = id's for running hyperparameter optimization
 */


// ---------------------------------------------------------------------------------------------------------------------
// Configs for hpopt will be a vector of ConfigMap types. Typedefs + helper functions for actual configs:
// ---------------------------------------------------------------------------------------------------------------------

// Type aliases
using ConfigValueRange = std::variant<std::string, bool, int, std::pair<double,double>>;
using HpoptConfigMap   = std::unordered_map<std::string, ConfigValueRange>;

// Templated Helper to read value from config map
template<typename T>
T get_config_value(const HpoptConfigMap& config, const std::string& key)
{
    if (!config.contains(key)) {
        std::stringstream err_msg;
        err_msg << "Expecting to find key (" << key << ") in ConfigMap, but couldn't.";
        throw std::runtime_error(err_msg.str());
    }

    // If wrong type, std::get will throw std::bad_variant_access
    return std::get<T>(config.at(key));
}


// ---------------------------------------------------------------------------------------------------------------------
// For hyperparam optimisation: set of integer parameters
// ---------------------------------------------------------------------------------------------------------------------

static const std::unordered_set<std::string> HPOPT_INTEGER_ALG_PARAM_IDS =
{
    ALG_PARAM_ID_UCT_BUDGET,
};

// ---------------------------------------------------------------------------------------------------------------------
// For hyperparam optimisation: set of parameters to search over a log scaling
// ---------------------------------------------------------------------------------------------------------------------

static const std::unordered_set<std::string> HPOPT_LOG_SCALE_ALG_PARAM_IDS =
{
    ALG_PARAM_ID_BIAS,
    ALG_PARAM_ID_UCT_BUDGET,
    ALG_PARAM_ID_INIT_TEMP,
    ALG_PARAM_ID_TEMP_DECAY_RATE,
    ALG_PARAM_ID_INIT_ENTROPY_COEFF,
    ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,
    ALG_PARAM_ID_EPSILON,
};
