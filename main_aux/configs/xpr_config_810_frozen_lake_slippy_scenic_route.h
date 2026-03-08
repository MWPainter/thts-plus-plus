#pragma once

#include "main_aux/configs/config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_810 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                     XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                     "810_supp_frozen_lake_slippy_scenic_route_4x4"},
        {XPR_PARAM_ID_ENV,                      ENV_ID_SLIPPY_FROZEN_LAKE_S_4x4},
        {XPR_PARAM_ID_MCTS_MODE,                false},
        {XPR_PARAM_ID_GRAPH_SEARCH,             true},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,         100},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_GLOBAL,  0.0},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_LOCAL,   1.0},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,          false},
        {XPR_PARAM_ID_TERMINATION_BOUND,        100000},
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,    10},
        {XPR_PARAM_ID_SEARCH_THREADS,           16},
        {XPR_PARAM_ID_EVAL_DELTA,               2500},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,            256},
        {XPR_PARAM_ID_EVAL_THREADS,             16},
    },

    //------------------------------------------------------------------------------------------------
    // range of ments params
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.001},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.003},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.01},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.03},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.1},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.3},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        1.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        3.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        10.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        30.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        100.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },

    //------------------------------------------------------------------------------------------------
    // range of rents params
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.001},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.003},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.01},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.03},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.1},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.3},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        1.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        3.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        10.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        30.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        100.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },


    //------------------------------------------------------------------------------------------------
    // range of tents params
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.001},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.003},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.01},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.03},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.1},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.3},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        1.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        3.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        10.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        30.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        100.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },


    //------------------------------------------------------------------------------------------------
    // range of bts params
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.001},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.003},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.01},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.03},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.1},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.3},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        1.0},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        3.0},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        10.0},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        30.0},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        100.0},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
    },
    

    //------------------------------------------------------------------------------------------------
    // range of dents params
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.001},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       0.001},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    500.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.003},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       0.003},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    500.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.01},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       0.01},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    500.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.03},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       0.03},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    500.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.1},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       0.1},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    500.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.3},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       0.3},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    500.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,        1.0},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       1.0},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    500.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,        3.0},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       3.0},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    500.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,        10.0},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       10.0},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    500.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,        30.0},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       30.0},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    500.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,        30.0},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       30.0},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    500.0},
    },
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,        100.0},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0},
        {ALG_PARAM_ID_EPSILON,          0.5},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  1.0},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       100.0},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    500.0},
    },
};
