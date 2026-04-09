#pragma once

#include "main_aux/configs/config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_470 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                     XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                     "470_eval_frozen_lake_sparse_8x8_heuristic_mcts_graph"},
        {XPR_PARAM_ID_ENV,                      ENV_ID_FROZEN_LAKE_S_8x8},
        {XPR_PARAM_ID_MCTS_MODE,                true},
        {XPR_PARAM_ID_GRAPH_SEARCH,             true},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,         100},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_GLOBAL,  1.0},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_LOCAL,   0.0},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,          false},
        {XPR_PARAM_ID_TERMINATION_BOUND,        100000}, //500000},
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,    100},
        {XPR_PARAM_ID_SEARCH_THREADS,           16},
        {XPR_PARAM_ID_EVAL_DELTA,               200},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,            512},
        {XPR_PARAM_ID_EVAL_THREADS,             16},
    },
    // uct params - 0.824832
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS,             4.75195},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // max uct params: 0.827154
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MAX_UCT},
        {ALG_PARAM_ID_BIAS,             5.1844},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // hmcts params: 0.835534
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_HMCTS},
        {ALG_PARAM_ID_BIAS,             4.61929},
        {ALG_PARAM_ID_UCT_BUDGET,       99975},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // ments params: 0.815162
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        1e-05},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          1.0},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_NORMALISE_Q,      0},
    },
    // rents params: 0.832094
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.000678582},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.561974},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_NORMALISE_Q,      0},
    },
    // tents params: 0.826494
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.000154342},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          1.0},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_NORMALISE_Q,      0},
    },
    // bts params: 0.819131
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        99.9736},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  99.9945},
        {ALG_PARAM_ID_EPSILON,          7.84088e-06},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // dents params: 0.813935
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                3.41238e-05},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          0.15104},
        {ALG_PARAM_ID_EPSILON,                  0.991613},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       1e-05},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    1.0e12},
        {ALG_PARAM_ID_HEURISTIC_VALUE,          1.0},
    },
};
