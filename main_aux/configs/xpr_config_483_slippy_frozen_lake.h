#pragma once

#include "main_aux/configs/config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_483 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                     XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                     "483_eval_slippy_frozen_lake_4x16_heuristic"},
        {XPR_PARAM_ID_ENV,                      ENV_ID_SLIPPY_FROZEN_LAKE_S_4x16},
        {XPR_PARAM_ID_MCTS_MODE,                false},
        {XPR_PARAM_ID_GRAPH_SEARCH,             true},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,         200},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_GLOBAL,  0.0},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_LOCAL,   1.0},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,          false},
        {XPR_PARAM_ID_TERMINATION_BOUND,        500000},
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,    35},
        {XPR_PARAM_ID_SEARCH_THREADS,           16},
        {XPR_PARAM_ID_EVAL_DELTA,               1000},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,            1024},
        {XPR_PARAM_ID_EVAL_THREADS,             16},
    },
    // uct params - 0.393652
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS,             7.07031},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // max uct params: 0.682715
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MAX_UCT},
        {ALG_PARAM_ID_BIAS,             2.58248},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // hmcts params: 0.401367
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_HMCTS},
        {ALG_PARAM_ID_BIAS,             6.81292},
        {ALG_PARAM_ID_UCT_BUDGET,       28},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // ments params: 0.74707
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.0102388},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.248039},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_NORMALISE_Q,      0},
    },
    // rents params: 0.744043
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.00288887},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.432172},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_NORMALISE_Q,      0},
    },
    // tents params: 0.746191
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.0100192},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.28869},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_NORMALISE_Q,      0},
    },
    // bts params: 0.739941
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.134244},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0482089},
        {ALG_PARAM_ID_EPSILON,          0.459669},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    // dents params: 0.745996
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                0.264239},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          0.199595},
        {ALG_PARAM_ID_EPSILON,                  0.443917},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       0.000917463},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    13838.7},
        {ALG_PARAM_ID_HEURISTIC_VALUE,          1.0},
    },
};
