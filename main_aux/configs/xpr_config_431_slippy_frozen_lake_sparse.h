#pragma once

#include "main_aux/configs/config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_431 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                     XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                     "431_eval_slippy_frozen_lake_sparse_5x5"},
        {XPR_PARAM_ID_ENV,                      ENV_ID_SLIPPY_FROZEN_LAKE_S_5x5},
        {XPR_PARAM_ID_MCTS_MODE,                true},
        {XPR_PARAM_ID_GRAPH_SEARCH,             false},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,         100},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,          false},
        {XPR_PARAM_ID_TERMINATION_BOUND,        250000},
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,    25},
        {XPR_PARAM_ID_SEARCH_THREADS,           16},
        {XPR_PARAM_ID_EVAL_DELTA,               500},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,            1024},
        {XPR_PARAM_ID_EVAL_THREADS,             16},
    },
    // uct params - 0.205301
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS, 5.14271},
    },
    // max uct params: 0.411523
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_MAX_UCT},
        {ALG_PARAM_ID_BIAS, 1.71339},
    },
    // hmcts params: 0.201668
    {
        {XPR_OR_ALG_ID_TAG,         ALG_ID_HMCTS},
        {ALG_PARAM_ID_BIAS,         4.26182},
        {ALG_PARAM_ID_UCT_BUDGET,   1},
    },
    // ments params: 0.530895
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.000175853},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,  1.63015},
        {ALG_PARAM_ID_EPSILON,          0.799133},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  0.0},
    },
    // rents params: 0.555237
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.130611},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,  2.23591},
        {ALG_PARAM_ID_EPSILON,          0.768541},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  0.0},
    },
    // tents params: 0.560449
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.111505},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0708811},
        {ALG_PARAM_ID_EPSILON,          0.857039},        
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  0.0},

    },
    // bts params: 0.527699
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.00302707},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  156.512},
        {ALG_PARAM_ID_EPSILON,          0.698257},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  0.0},
    },
    // dents params: 0.555859
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                0.00109236},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          0.000168291},
        {ALG_PARAM_ID_EPSILON,                  0.602529},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,          0.0},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       8.19793},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    50.542},
    },
};
