#pragma once

#include "main_aux/configs/config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_400 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                     XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                     "400_eval_frozen_lake_dense_8x8"},
        {XPR_PARAM_ID_ENV,                      ENV_ID_FROZEN_LAKE_D_8x8},
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
    // uct params - -22.3
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS, 0.00535062},
    },
    // max uct params: -17.5
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_MAX_UCT},
        {ALG_PARAM_ID_BIAS, 2.46939},
    },
    // hmcts params: -20.7
    {
        {XPR_OR_ALG_ID_TAG,         ALG_ID_HMCTS},
        {ALG_PARAM_ID_BIAS,         0.00761479},
        {ALG_PARAM_ID_UCT_BUDGET,   1},
    },
    // ments params: 19.2
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.000504033},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,,  0.0001},
        {ALG_PARAM_ID_EPSILON,          1.0},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  -100.0},
    },
    // rents params: -19.2
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.000100067},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,,  16.1997},
        {ALG_PARAM_ID_EPSILON,          0.999589},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  -99.9981},
    },
    // tents params: -19.3
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.000100172},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,,  0.0548275},
        {ALG_PARAM_ID_EPSILON,          0.999662},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  -99.9903},
    },
    // bts params: -19.6
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        217.301},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  9998.4},
        {ALG_PARAM_ID_EPSILON,          0.999992},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  -0.00214427},
    },
    // dents params: -18.6
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                144.603},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          9993.69},
        {ALG_PARAM_ID_EPSILON,                  0.999533},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,          -0.00324917},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       0.010019},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    784.985},
    },
};
