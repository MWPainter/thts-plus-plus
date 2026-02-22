#pragma once

#include "main_aux/configs/config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_002 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                     XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                     "002_debug"},
        {XPR_PARAM_ID_ENV,                      ENV_ID_SAILING_NORTH_ID},
        {XPR_PARAM_ID_MCTS_MODE,                true},
        {XPR_PARAM_ID_GRAPH_SEARCH,             true},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,         100},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,          false},
        {XPR_PARAM_ID_TERMINATION_BOUND,        2},
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,    1},
        {XPR_PARAM_ID_SEARCH_THREADS,           1},
        {XPR_PARAM_ID_EVAL_DELTA,               1},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,            1},
        {XPR_PARAM_ID_EVAL_THREADS,             1},
    },
    // uct params
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS, 1000.0},
        {ALG_PARAM_ID_HEURISTIC_VALUE, 0.0},
    },
};