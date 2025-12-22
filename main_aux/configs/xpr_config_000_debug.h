#pragma once

#include "main_aux/configs/config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"

// static const std::vector<ConfigMap> CONFIG_000 =
// {
//     // xpr params
//     {
//         {XPR_OR_ALG_ID_TAG,                     XPR_PARAMS_ID_TAG},
//         {XPR_PARAM_ID_NAME,                     "000_debug"},
//         {XPR_PARAM_ID_ENV,                      ENV_ID_D_CHAIN_10},
//         {XPR_PARAM_ID_MCTS_MODE,                true},
//         {XPR_PARAM_ID_GRAPH_SEARCH,             false},
//         {XPR_PARAM_ID_MAX_TRIAL_LENGTH,         10000},
//         {XPR_PARAM_ID_RUNTIME_BOUNDED,          false},
//         {XPR_PARAM_ID_TERMINATION_BOUND,        100.0},
//         {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,    2},
//         {XPR_PARAM_ID_SEARCH_THREADS,           2},
//         {XPR_PARAM_ID_EVAL_DELTA,               25},
//         {XPR_PARAM_ID_EVAL_ROLLOUTS,            25},
//         {XPR_PARAM_ID_EVAL_THREADS,             2},
//     },
//     // uct params
//     {
//         {XPR_OR_ALG_ID_TAG, ALG_ID_UCT},
//         {ALG_PARAM_ID_BIAS, 4.0},
//     },
//     // max uct params
//     {
//         {XPR_OR_ALG_ID_TAG, ALG_ID_MAX_UCT},
//         {ALG_PARAM_ID_BIAS, 4.0},
//     },
//     // hmcts params
//     {
//         {XPR_OR_ALG_ID_TAG,         ALG_ID_HMCTS},
//         {ALG_PARAM_ID_BIAS,         4.0},
//         {ALG_PARAM_ID_UCT_BUDGET,   10},
//     },
//     // ments params
//     {
//         {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
//         {ALG_PARAM_ID_INIT_TEMP,        1.0},
//         {ALG_PARAM_ID_TEMP_DECAY_RATE,  1.0},
//         {ALG_PARAM_ID_EPSILON,          0.1},
//         {ALG_PARAM_ID_DEFAULT_Q_VALUE,  0.0},
//     },
//     // rents params
//     {
//         {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
//         {ALG_PARAM_ID_INIT_TEMP,        1.0},
//         {ALG_PARAM_ID_TEMP_DECAY_RATE,  1.0},
//         {ALG_PARAM_ID_EPSILON,          0.1},
//         {ALG_PARAM_ID_DEFAULT_Q_VALUE,  0.0},
//     },
//     // tents params
//     {
//         {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
//         {ALG_PARAM_ID_INIT_TEMP,        1.0},
//         {ALG_PARAM_ID_TEMP_DECAY_RATE,  1.0},
//         {ALG_PARAM_ID_EPSILON,          0.1},
//         {ALG_PARAM_ID_DEFAULT_Q_VALUE,  0.0},
//     },
//     // bts params
//     {
//         {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
//         {ALG_PARAM_ID_INIT_TEMP,        1.0},
//         {ALG_PARAM_ID_TEMP_DECAY_RATE,  1.0},
//         {ALG_PARAM_ID_EPSILON,          0.1},
//         {ALG_PARAM_ID_DEFAULT_Q_VALUE,  0.0},
//     },
//     // dents params
//     {
//         {XPR_OR_ALG_ID_TAG,                     ALG_ID_DENTS},
//         {ALG_PARAM_ID_INIT_TEMP,                1.0},
//         {ALG_PARAM_ID_TEMP_DECAY_RATE,          1.0},
//         {ALG_PARAM_ID_EPSILON,                  0.1},
//         {ALG_PARAM_ID_DEFAULT_Q_VALUE,          0.0},
//         {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       1.0},
//         {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    1.0},
//     },
// };




static const std::vector<ConfigMap> CONFIG_000 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                     XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                     "000_debug"},
        {XPR_PARAM_ID_ENV,                      ENV_ID_FROZEN_LAKE_D_8x16},
        {XPR_PARAM_ID_MCTS_MODE,                false},
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
    // max uct params: -17.5
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_MAX_UCT},
        {ALG_PARAM_ID_BIAS, 2.46939},
    },
};
