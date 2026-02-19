#pragma once

#include "main_aux/configs/config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_452 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                     XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                     "452_eval_sailing_south_east_16x16"},
        {XPR_PARAM_ID_ENV,                      ENV_ID_SAILING_16x16_SOUTH_EAST_ID},
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
    // uct params - -226.204
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS, 0.0883178},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
    },
    // max uct params: -224.499
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_MAX_UCT},
        {ALG_PARAM_ID_BIAS, 0.389263},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
    },
    // hmcts params: -225.882
    {
        {XPR_OR_ALG_ID_TAG,         ALG_ID_HMCTS},
        {ALG_PARAM_ID_BIAS,         0.0967555},
        {ALG_PARAM_ID_UCT_BUDGET,   9},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  0.0},
    },
    // ments params: -226.093
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.0473387},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,  16.8901},
        {ALG_PARAM_ID_EPSILON,          0.263028},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  -90.53},
    },
    // rents params: -227.238
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.0475893},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,  12.0975},
        {ALG_PARAM_ID_EPSILON,          0.999743},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  -89.5957},
    },
    // tents params: -226.635
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.10757},
        // {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0164081},
        {ALG_PARAM_ID_EPSILON,          1.0},        
        {ALG_PARAM_ID_HEURISTIC_VALUE,  -99.994},

    },
    // bts params: -224.742
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.2646},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.144703},
        {ALG_PARAM_ID_EPSILON,          0.162603},
        {ALG_PARAM_ID_HEURISTIC_VALUE,  -42.7671},
    },
    // dents params: -225.263
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                0.188999},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          0.296287},
        {ALG_PARAM_ID_EPSILON,                  0.0746498},
        {ALG_PARAM_ID_HEURISTIC_VALUE,          -0.0274888},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       0.000100453},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    1.00086},
    },
};
