#pragma once

#include "main_aux/configs/config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_014 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                     XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                     "014_sailing_debugging"},
        {XPR_PARAM_ID_ENV,                      ENV_ID_SAILING_NORTH_ID},
        {XPR_PARAM_ID_MCTS_MODE,                true},
        {XPR_PARAM_ID_GRAPH_SEARCH,             true},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,         100},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_GLOBAL,  1.0},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_LOCAL,   0.0},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,          false},
        {XPR_PARAM_ID_TERMINATION_BOUND,        100000},
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,    10},
        {XPR_PARAM_ID_SEARCH_THREADS,           16},
        {XPR_PARAM_ID_EVAL_DELTA,               2500},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,            256},
        {XPR_PARAM_ID_EVAL_THREADS,             16},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        100.0},
        {ALG_PARAM_ID_EPSILON,          0.0},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        1000.0},
        {ALG_PARAM_ID_EPSILON,          0.0},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
    },
};
