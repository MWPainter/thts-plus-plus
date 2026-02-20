#pragma once

#include "main_aux/configs/config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_402 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                     XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                     "402_eval_frozen_lake_dense_16x16"},
        {XPR_PARAM_ID_ENV,                      ENV_ID_FROZEN_LAKE_D_16x16},
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
    // uct params - -15
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS, 0.00662908},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
    },
    // max uct params: -21.7
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_MAX_UCT},
        {ALG_PARAM_ID_BIAS, 8.56652},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
    },
    // hmcts params: -15.6
    {
        {XPR_OR_ALG_ID_TAG,         ALG_ID_HMCTS},
        {ALG_PARAM_ID_BIAS,         0.00433374},
        {ALG_PARAM_ID_UCT_BUDGET,   6},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
    },
    // ments params: -18.3
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.000100165},
        {ALG_PARAM_ID_EPSILON,          1.0},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
        // {ALG_PARAM_ID_HEURISTIC_VALUE,  -12.3553},
    },
    // rents params: -20.8
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.000380742},
        {ALG_PARAM_ID_EPSILON,          1.0},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
        // {ALG_PARAM_ID_HEURISTIC_VALUE,  -83.5272},
    },
    // tents params: -18.4
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.000100176},
        {ALG_PARAM_ID_EPSILON,          1.0},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
        // {ALG_PARAM_ID_HEURISTIC_VALUE,  -85.717},
    },
    // bts params: -15
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,                0.999911},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          0.351609},
        // {ALG_PARAM_ID_INIT_TEMP,        98.2299},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,  779.454},
        {ALG_PARAM_ID_EPSILON,          1.0},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
    },
    // dents params: -15
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                0.999911},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          0.351609},
        {ALG_PARAM_ID_EPSILON,                  1.0},
        {ALG_PARAM_ID_HEURISTIC_VALUE,          0.0},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       0.000365626},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    10000.0},
    },
};
