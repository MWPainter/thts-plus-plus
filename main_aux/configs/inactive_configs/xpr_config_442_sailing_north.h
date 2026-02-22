#pragma once

#include "main_aux/configs/config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_442 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                     XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                     "442_eval_sailing_north_16x16"},
        {XPR_PARAM_ID_ENV,                      ENV_ID_SAILING_16x16_NORTH_ID},
        {XPR_PARAM_ID_MCTS_MODE,                true},
        {XPR_PARAM_ID_GRAPH_SEARCH,             false},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,         100},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,          false},
        {XPR_PARAM_ID_TERMINATION_BOUND,        250000},
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,    25},
        {XPR_PARAM_ID_SEARCH_THREADS,           16},
        {XPR_PARAM_ID_EVAL_DELTA,               1000},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,            512},
        {XPR_PARAM_ID_EVAL_THREADS,             16},
    },
    // uct params - -222.779
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS, 0.0857737},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
    },
    // max uct params: -220.874
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_MAX_UCT},
        {ALG_PARAM_ID_BIAS, 0.470522},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
    },
    // hmcts params: -222.461
    {
        {XPR_OR_ALG_ID_TAG,         ALG_ID_HMCTS},
        {ALG_PARAM_ID_BIAS,         0.110371},
        {ALG_PARAM_ID_UCT_BUDGET,   6},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
    },
    // ments params: -222.613
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.000100003},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,  16.8901},
        {ALG_PARAM_ID_EPSILON,          0.279416},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  -99.995},
    },
    // rents params: -222.987
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        1.29657},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,  12.0975},
        {ALG_PARAM_ID_EPSILON,          1.00045e-06},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  -32.3045},
    },
    // tents params: -223.371
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.12976},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0164081},
        {ALG_PARAM_ID_EPSILON,          0.99659},        
        {ALG_PARAM_ID_HEURISTIC_VALUE,  -100.0},

    },
    // bts params: -221.573
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        1.56971},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.413781},
        {ALG_PARAM_ID_EPSILON,          0.112603},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  -0.0211122},
    },
    // dents params: -221.499
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                1.25388},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          0.888495},
        {ALG_PARAM_ID_EPSILON,                  0.124649},
        {ALG_PARAM_ID_HEURISTIC_VALUE,          -0.00180721},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       0.000100159},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    1.00086},
    },
};
