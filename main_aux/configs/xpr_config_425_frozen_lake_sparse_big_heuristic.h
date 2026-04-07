#pragma once

#include "main_aux/configs/config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_425 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                     XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                     "425_eval_frozen_lake_sparse_8x32_heuristic"},
        {XPR_PARAM_ID_ENV,                      ENV_ID_FROZEN_LAKE_S_8x32},
        {XPR_PARAM_ID_MCTS_MODE,                false},
        {XPR_PARAM_ID_GRAPH_SEARCH,             true},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,         200},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_GLOBAL,  0.0},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_LOCAL,   1.0},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,          false},
        {XPR_PARAM_ID_TERMINATION_BOUND,        2000000},
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,    25},
        {XPR_PARAM_ID_SEARCH_THREADS,           16},
        {XPR_PARAM_ID_EVAL_DELTA,               5000},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,            512},
        {XPR_PARAM_ID_EVAL_THREADS,             16},
    },
    // uct params - 0.753017
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS,             7.00903},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // max uct params: 0.761017
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MAX_UCT},
        {ALG_PARAM_ID_BIAS,             6.86761},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // hmcts params: 0.746872
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_HMCTS},
        {ALG_PARAM_ID_BIAS,             1.50717},
        {ALG_PARAM_ID_UCT_BUDGET,       99564},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
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
    // rents params: 0.793614
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.000134244},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          1.0},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_NORMALISE_Q,      0},
    },
    // tents params: 0.793614
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.0000920491},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.743629},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_NORMALISE_Q,      0},
    },
    // bts params: 0.793614
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        1.00158e-05},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  3.16652},
        {ALG_PARAM_ID_EPSILON,          0.999702},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // dents params: 0.793614
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                0.00404148},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          0.9773},
        {ALG_PARAM_ID_EPSILON,                  0.743917},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       0.000335841},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    1.0e12},
        {ALG_PARAM_ID_HEURISTIC_VALUE,          1.0},
    },
};
