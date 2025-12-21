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
        {XPR_PARAM_ID_EVAL_DELTA,               500},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,            1024},
        {XPR_PARAM_ID_EVAL_THREADS,             16},
    },
    // uct params - -222.146
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS, 0.096123},
    },
    // max uct params: -221.204
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_MAX_UCT},
        {ALG_PARAM_ID_BIAS, 0.373008},
    },
    // hmcts params: -221.563
    {
        {XPR_OR_ALG_ID_TAG,         ALG_ID_HMCTS},
        {ALG_PARAM_ID_BIAS,         0.0926668},
        {ALG_PARAM_ID_UCT_BUDGET,   6},
    },
    // ments params: -220.449
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        6.24637},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  16.8901},
        {ALG_PARAM_ID_EPSILON,          0.176438},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  -122.977},
    },
    // rents params: -219.884
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        3.47742},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  12.0975},
        {ALG_PARAM_ID_EPSILON,          0.252338},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  -120.505},
    },
    // tents params: 0.823497
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        0.0848362},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.0164081},
        {ALG_PARAM_ID_EPSILON,          0.348206},        
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  -148.039},

    },
    // bts params: -221.573
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        1.56971},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  0.413781},
        {ALG_PARAM_ID_EPSILON,          0.112603},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  -0.0211122},
    },
    // dents params: -221.499
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                253.808},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          9996.88},
        {ALG_PARAM_ID_EPSILON,                  0.000746498},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,          -173.041},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       0.000100159},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    1.00086},
    },
};
