#pragma once

#include "main_aux/configs/config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_432 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                     XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                     "432_eval_slippy_frozen_lake_sparse_6x6"},
        {XPR_PARAM_ID_ENV,                      ENV_ID_SLIPPY_FROZEN_LAKE_S_6x6},
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
    // uct params - 0.667676
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS, 0.312918},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // max uct params: 0.419649
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_MAX_UCT},
        {ALG_PARAM_ID_BIAS, 1.73595},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // hmcts params: 0.667676
    {
        {XPR_OR_ALG_ID_TAG,         ALG_ID_HMCTS},
        {ALG_PARAM_ID_BIAS, 0.312918},
        {ALG_PARAM_ID_UCT_BUDGET,   3109},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // ments params: 0.227606
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.00183624},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,  1.63015},
        {ALG_PARAM_ID_EPSILON,          1.0},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // rents params: 0.372373
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.000100449},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,  2.23591},
        {ALG_PARAM_ID_EPSILON,          0.917492},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // tents params: 0.238136
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.000146918},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0708811},
        {ALG_PARAM_ID_EPSILON,          0.9997},        
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},

    },
    // bts params: 0.705849
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.00302707},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  156.512},
        {ALG_PARAM_ID_EPSILON,          0.140096},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // dents params: 0.695312
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                0.00193928},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          0.0964109},
        {ALG_PARAM_ID_EPSILON,                  0.0602529},
        {ALG_PARAM_ID_HEURISTIC_VALUE,          1.0},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       0.10026},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    250.542},
    },
};
