#pragma once

#include "main_aux/configs/config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_410 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                     XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                     "410_eval_frozen_lake_sparse_8x8"},
        {XPR_PARAM_ID_ENV,                      ENV_ID_FROZEN_LAKE_S_8x8},
        {XPR_PARAM_ID_MCTS_MODE,                false},
        {XPR_PARAM_ID_GRAPH_SEARCH,             true},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,         100},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,          false},
        {XPR_PARAM_ID_TERMINATION_BOUND,        250000},
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,    25},
        {XPR_PARAM_ID_SEARCH_THREADS,           16},
        {XPR_PARAM_ID_EVAL_DELTA,               1000},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,            512},
        {XPR_PARAM_ID_EVAL_THREADS,             16},
    },
    // uct params - 0.835016
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS, 3.10925},
    },
    // max uct params: 0.794887
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_MAX_UCT},
        {ALG_PARAM_ID_BIAS, 1.42329},
    },
    // hmcts params: 0.83045
    {
        {XPR_OR_ALG_ID_TAG,         ALG_ID_HMCTS},
        {ALG_PARAM_ID_BIAS,         3.16228},
        {ALG_PARAM_ID_UCT_BUDGET,   5},
    },
    // ments params: 0.827363
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.0026165},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.00284344},
        {ALG_PARAM_ID_EPSILON,          1.0},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  0.0},
    },
    // rents params: 0.824824
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.000141045},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.00010024},
        {ALG_PARAM_ID_EPSILON,          1.0},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  0.0},
    },
    // tents params: 0.830534
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.00151683},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0001},
        {ALG_PARAM_ID_EPSILON,          1.0},      
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  0.0},

    },
    // bts params: 0.830997
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.540259},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.00843341},
        {ALG_PARAM_ID_EPSILON,          0.999823},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  0.0},
    },
    // dents params: 0.839783
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                0.897669,},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          0.000339204},
        {ALG_PARAM_ID_EPSILON,                  0.999702},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,          0.0},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       0.00114433},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    82.446},
    },
};
