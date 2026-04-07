#pragma once

#include "main_aux/configs/config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_490 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                     XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                     "490_eval_sailing_north_8x8_heuristic"},
        {XPR_PARAM_ID_ENV,                      ENV_ID_SAILING_NORTH_ID},
        {XPR_PARAM_ID_MCTS_MODE,                false},
        {XPR_PARAM_ID_GRAPH_SEARCH,             true},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,         100},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_GLOBAL,  0.0},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_LOCAL,   1.0},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,          false},
        {XPR_PARAM_ID_TERMINATION_BOUND,        100000},
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,    35},
        {XPR_PARAM_ID_SEARCH_THREADS,           16},
        {XPR_PARAM_ID_EVAL_DELTA,               1000},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,            512},
        {XPR_PARAM_ID_EVAL_THREADS,             16},
    },
    // uct params - -19.0935
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS,             0.0948255},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
    },
    // max uct params: -18.0437
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MAX_UCT},
        {ALG_PARAM_ID_BIAS,             0.0905458},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
    },
    // hmcts params: -18.483
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_HMCTS},
        {ALG_PARAM_ID_BIAS,             0.0810641},
        {ALG_PARAM_ID_UCT_BUDGET,       32},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
    },
    // ments params: -16.8836
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.000539729},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.999937},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
        {ALG_PARAM_ID_NORMALISE_Q,      1},
    },
    // rents params: -16.8461
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.000157324},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.319173},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
        {ALG_PARAM_ID_NORMALISE_Q,      1},
    },
    // tents params: -16.7279
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.00076065},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.999739},
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
    // dents params: -17.1915
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                1.01505e-05},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          0.813025},
        {ALG_PARAM_ID_EPSILON,                  0.971033},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       0.00114117},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    21857.8},
        {ALG_PARAM_ID_HEURISTIC_VALUE,          0.0},
    },
};
