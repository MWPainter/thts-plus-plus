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
    // uct params - -24.7
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS, 0.00749459},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
    },
    // max uct params: -19.5
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_MAX_UCT},
        {ALG_PARAM_ID_BIAS, 2.16406},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
    },
    // hmcts params: -24.2
    {
        {XPR_OR_ALG_ID_TAG,         ALG_ID_HMCTS},
        {ALG_PARAM_ID_BIAS,         0.00695233},
        {ALG_PARAM_ID_UCT_BUDGET,   1},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
    },
    // ments params: -18.6364
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.173716},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,,  0.0001},
        {ALG_PARAM_ID_EPSILON,          0.999769},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  -88.1886},
    },
    // rents params: -21.5
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.609421},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,,  16.1997},
        {ALG_PARAM_ID_EPSILON,          6.45898e-06},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  -93.2885},
    },
    // tents params: -19.5
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.288648},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,,  0.0548275},
        {ALG_PARAM_ID_EPSILON,          0.999839},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  -87.412},
    },
    // bts params: -18.8
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.1606965},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  1.00254e-06},
        {ALG_PARAM_ID_EPSILON,          1.0},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  -0.0027583},
    },
    // dents params: -18.2
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                0.6363144},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          0.00224227},
        {ALG_PARAM_ID_EPSILON,                  0.999533},
        {ALG_PARAM_ID_HEURISTIC_VALUE,          -32.98},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       1.1001274},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    54.89},
    },
};
