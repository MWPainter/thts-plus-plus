#pragma once

#include "main_aux/configs/constants_envs.h"

#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include <stdexcept>

/**
 * This file defines XPR_PARAM_ID constants
 * - XPR_PARAM_ID = id's for parameters that each experiment should specify
 */


// ---------------------------------------------------------------------------------------------------------------------
// Tags to indicate what a config dictionary contains to indicate experiment level params in configs
// ---------------------------------------------------------------------------------------------------------------------

static const std::string XPR_OR_ALG_ID_TAG = "xpr_or_alg_id";                   // Special key to indicate if a dictonray is specifying xpr level params or alg level params
static const std::string XPR_PARAMS_ID_TAG = "xpr_params";                      // Special value to indicate dictionary is specifying xpr level params

// ---------------------------------------------------------------------------------------------------------------------
// Constants used in the tree search, but to be varied on a per experiment basis
// ---------------------------------------------------------------------------------------------------------------------

static const std::string XPR_PARAM_ID_NAME = "xpr_name";                        // user readable name for experiment
static const std::string XPR_PARAM_ID_ENV = "env_id";                           // the env id for this experiment
static const std::string XPR_PARAM_ID_MCTS_MODE = "mcts_mode";                  // if MCTS mode should be used
static const std::string XPR_PARAM_ID_GRAPH_SEARCH = "graph_search";            // if should run over graph instead of tree (transposition table use)
static const std::string XPR_PARAM_ID_MAX_TRIAL_LENGTH = "max_trial_length";    // max trial length
static const std::string XPR_PARAM_ID_RUNTIME_BOUNDED = "runtime_bounded";      // if algorithms should be bounded using runtime (or number of trials)
static const std::string XPR_PARAM_ID_TERMINATION_BOUND = "term_bound";         // runtime (or #trials) that algorithm is allowed
static const std::string XPR_PARAM_ID_REPEATED_RUNS_PER_ALG = "num_repeats";    // number of times to repeat running each algorithms
static const std::string XPR_PARAM_ID_SEARCH_THREADS = "search_threads";        // number of threads to use in search
static const std::string XPR_PARAM_ID_EVAL_DELTA = "eval_delta";                // delta (in runtime/#trials) to evaluate algorithms at
static const std::string XPR_PARAM_ID_EVAL_ROLLOUTS = "eval_rollouts";          // numbber of rollouts for each MC eval
static const std::string XPR_PARAM_ID_EVAL_THREADS = "eval_threads";            // number of threads to use in evaluation

// ---------------------------------------------------------------------------------------------------------------------
// Configs will be a vector of ConfigMap types. Typedefs + helper functions for actual configs:
// ---------------------------------------------------------------------------------------------------------------------

// Type aliases
using ConfigValue = std::variant<std::string, bool, int, double>;
using ConfigMap   = std::unordered_map<std::string, ConfigValue>;

// Templated Helper to read value from config map
template<typename T>
T get_config_value(const ConfigMap& config, const std::string& key)
{
    if (!config.contains(key)) {
        std::stringstream err_msg;
        err_msg << "Expecting to find key (" << key << ") in ConfigMap, but couldn't.";
        throw std::runtime_error(err_msg.str());
    }

    // If wrong type, std::get will throw std::bad_variant_access
    return std::get<T>(config.at(key));
}
