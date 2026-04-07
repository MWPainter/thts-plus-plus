#pragma once

#include "main_aux/configs/config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_501 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                     XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                     "501_eval_sailing_south_east_8x16_heuristic"},
        {XPR_PARAM_ID_ENV,                      ENV_ID_SAILING_8x16_SOUTH_EAST_ID},
        {XPR_PARAM_ID_MCTS_MODE,                false},
        {XPR_PARAM_ID_GRAPH_SEARCH,             true},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,         100},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_GLOBAL,  0.0},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_LOCAL,   1.0},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,          false},
        {XPR_PARAM_ID_TERMINATION_BOUND,        250000},
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,    35},
        {XPR_PARAM_ID_SEARCH_THREADS,           16},
        {XPR_PARAM_ID_EVAL_DELTA,               1000},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,            512},
        {XPR_PARAM_ID_EVAL_THREADS,             16},
    },
    // uct params - -23.3769
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS,             0.180558},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
    },
    // max uct params: -24.1624
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MAX_UCT},
        {ALG_PARAM_ID_BIAS,             0.22578},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
    },
    // hmcts params: -22.6347
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_HMCTS},
        {ALG_PARAM_ID_BIAS,             0.144804},
        {ALG_PARAM_ID_UCT_BUDGET,       100},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
    },
    // ments params: -21.3718
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.000605002},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          1.0},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
        {ALG_PARAM_ID_NORMALISE_Q,      1},
    },
    // rents params: -21.7517
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        1e-05},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          1.0},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
        {ALG_PARAM_ID_NORMALISE_Q,      1},
    },
    // tents params: -21.0504
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.00324456},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.999916},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
        {ALG_PARAM_ID_NORMALISE_Q,      1},
    },
    // bts params: -16.9756
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.00232156},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.00116672},
        {ALG_PARAM_ID_EPSILON,          0.999677},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
    },
    // dents params: -21.5108
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                1.0013e-05},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          9997.15},
        {ALG_PARAM_ID_EPSILON,                  0.999991},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       1.00121e-05},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    39123.6},
        {ALG_PARAM_ID_HEURISTIC_VALUE,          0.0},
    },
};
