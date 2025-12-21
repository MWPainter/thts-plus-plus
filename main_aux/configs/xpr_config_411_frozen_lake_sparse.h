#pragma once

#include "main_aux/configs/config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_411 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                     XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                     "411_eval_frozen_lake_sparse_8x16"},
        {XPR_PARAM_ID_ENV,                      ENV_ID_FROZEN_LAKE_S_8x16},
        {XPR_PARAM_ID_MCTS_MODE,                true},
        {XPR_PARAM_ID_GRAPH_SEARCH,             false},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,         100},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,          false},
        {XPR_PARAM_ID_TERMINATION_BOUND,        250000},
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,    25},
        {XPR_PARAM_ID_SEARCH_THREADS,           16},
        {XPR_PARAM_ID_EVAL_DELTA,               500},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,            1024},
        {XPR_PARAM_ID_EVAL_THREADS,             16},
    },
    // uct params - 0.830846
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS, 3.42101},
    },
    // max uct params: 0.819522
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_MAX_UCT},
        {ALG_PARAM_ID_BIAS, 1.99952},
    },
    // hmcts params: 0.83045
    {
        {XPR_OR_ALG_ID_TAG,         ALG_ID_HMCTS},
        {ALG_PARAM_ID_BIAS,         3.16228},
        {ALG_PARAM_ID_UCT_BUDGET,   5},
    },
    // ments params: 19.2
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.0001},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.00284344},
        {ALG_PARAM_ID_EPSILON,          1.0},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  0.0},
    },
    // rents params: 0.825869
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.0011585},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.00010024},
        {ALG_PARAM_ID_EPSILON,          0.999815},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  0.0},
    },
    // tents params: 0.823497
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.00604732},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0001},
        {ALG_PARAM_ID_EPSILON,          0.65416},        
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  0.0},

    },
    // bts params: 0.830768
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.540259},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.00843341},
        {ALG_PARAM_ID_EPSILON,          0.423088},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  0.0},
    },
    // dents params: 0.839783
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                29.2042},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          66.7969},
        {ALG_PARAM_ID_EPSILON,                  0.427961},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,          0.0},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       71.6231},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    382.446},
    },
};
