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
    // uct params - 0.85501
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS, 4.89981},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // max uct params: 0.823429
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_MAX_UCT},
        {ALG_PARAM_ID_BIAS, 2.08546},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // hmcts params: 0.837212
    {
        {XPR_OR_ALG_ID_TAG,         ALG_ID_HMCTS},
        {ALG_PARAM_ID_BIAS,         0.573555},
        {ALG_PARAM_ID_UCT_BUDGET,   4999},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // ments params: 0.208079
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.00143526},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.00284344},
        {ALG_PARAM_ID_EPSILON,          1.0},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // rents params: 0.854326
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.0142845},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.00010024},
        {ALG_PARAM_ID_EPSILON,          1.0},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // tents params: 0.13828
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.00662414},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0001},
        {ALG_PARAM_ID_EPSILON,          1.0},      
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},

    },
    // bts params: 0.832193
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.0100064},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0843341},
        {ALG_PARAM_ID_EPSILON,          1.0},   
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // dents params: 0.805079
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.0100064},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0843341},
        {ALG_PARAM_ID_EPSILON,          1.0},   
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       0.051004},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    100.043},
    },
};
