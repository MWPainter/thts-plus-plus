#pragma once

#include "main_aux/configs/config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_101 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                     XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                     "101_intro_ssp_gridworld"},
        {XPR_PARAM_ID_ENV,                      ENV_ID_FROZEN_LAKE_NO_HOLE_DENSE},
        {XPR_PARAM_ID_MCTS_MODE,                false},
        {XPR_PARAM_ID_GRAPH_SEARCH,             false},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,         36},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_GLOBAL,  0.0},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_LOCAL,   0.0},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,          false},
        {XPR_PARAM_ID_TERMINATION_BOUND,        2500},
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,    100},
        {XPR_PARAM_ID_SEARCH_THREADS,           16},
        {XPR_PARAM_ID_EVAL_DELTA,               50},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,            16},
        {XPR_PARAM_ID_EVAL_THREADS,             16},
    },
    // uct params
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS,             10.0},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0}, // heuristic value == deault q value in this xpr
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS,             0.1},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0}, // heuristic value == deault q value in this xpr
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS,             0.01},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0}, // heuristic value == deault q value in this xpr
    },
    // ments params
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,                10.0},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          0.0},
        {ALG_PARAM_ID_EPSILON,                  0.0},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_NORMALISE_Q,      1},
    },
    // bts params
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,                10.0},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          0.0},
        {ALG_PARAM_ID_EPSILON,                  0.0},
        {ALG_PARAM_ID_HEURISTIC_VALUE,          1.0},
    },
    // dents params
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                10.0},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       10.0},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          0.0},
        {ALG_PARAM_ID_EPSILON,                  0.0},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    1000000.0},
        {ALG_PARAM_ID_HEURISTIC_VALUE,          1.0},
    },
};
