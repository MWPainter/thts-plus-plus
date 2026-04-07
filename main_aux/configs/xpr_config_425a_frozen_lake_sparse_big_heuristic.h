#pragma once

#include "main_aux/configs/config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_425a =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                     XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                     "425a_eval_frozen_lake_sparse_8x32_heuristic"},
        {XPR_PARAM_ID_ENV,                      ENV_ID_FROZEN_LAKE_S_8x32},
        {XPR_PARAM_ID_MCTS_MODE,                false},
        {XPR_PARAM_ID_GRAPH_SEARCH,             true},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,         200},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_GLOBAL,  0.0},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_LOCAL,   1.0},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,          false},
        {XPR_PARAM_ID_TERMINATION_BOUND,        5000000},
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,    25},
        {XPR_PARAM_ID_SEARCH_THREADS,           16},
        {XPR_PARAM_ID_EVAL_DELTA,               5000},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,            512},
        {XPR_PARAM_ID_EVAL_THREADS,             16},
    },
    // ments params: 0.793614
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.000681292},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          1.0},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_NORMALISE_Q,      0},
    },
};
