#pragma once

#include "main_aux/configs/config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_999 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                     XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                     "999_ments_entropy_trap_debug"},
        {XPR_PARAM_ID_ENV,                      ENV_ID_ENTROPY_TRAP_15},
        {XPR_PARAM_ID_MCTS_MODE,                false},
        {XPR_PARAM_ID_GRAPH_SEARCH,             false},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,         10000},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,          false},
        {XPR_PARAM_ID_TERMINATION_BOUND,        10000.0},
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,    25},
        {XPR_PARAM_ID_SEARCH_THREADS,           8},
        {XPR_PARAM_ID_EVAL_DELTA,               50},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,            1}, // det env
        {XPR_PARAM_ID_EVAL_THREADS,             1}, // det env
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        100.0},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.01},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  0.0},
    },
};
