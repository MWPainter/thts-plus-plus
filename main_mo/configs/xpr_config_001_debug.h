#pragma once

#include "main_mo/configs/config_map.h"

#include "main_mo/configs/constants_algorithms.h"
#include "main_mo/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_001 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                             XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                             "001_debug"},
        {XPR_PARAM_ID_ENV,                              ENV_ID_PY_DEBUG_2},
        {XPR_PARAM_ID_MCTS_MODE,                        true},
        {XPR_PARAM_ID_GRAPH_SEARCH,                     true},
        {XPR_PARAM_ID_VECTOR_VISIT_COUNTS,              false},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,                 10000},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,                  true},
        {XPR_PARAM_ID_TERMINATION_BOUND,                1.0},
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,            5},
        {XPR_PARAM_ID_SEARCH_THREADS,                   16},
        {XPR_PARAM_ID_EVAL_DELTA,                       0.25},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,                    128},
        {XPR_PARAM_ID_EVAL_THREADS,                     16},
        {XPR_PARAM_ID_CONVEX_HULL_MAX_SIZE,             -1},
        {XPR_PARAM_ID_CONVEX_HULL_TOLERANCE,            1e-9},
    },
    // czt params
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CZT},
        {ALG_PARAM_ID_BIAS,                         2.0},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  4},
    },
    // czt doubling params
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CZT_DOUBLING},
        {ALG_PARAM_ID_BIAS,                         2.0},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  4},
        {ALG_PARAM_ID_MIN_LOG2_N,                   3},
    },
    // ch uct params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_UCT},
        {ALG_PARAM_ID_BIAS,                 4.0},
    },
    // ch czt params
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CH_CZT},
        {ALG_PARAM_ID_BIAS,                         2.0},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  4},
    },
    // ch czt doubling params
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CH_CZT_DOUBLING},
        {ALG_PARAM_ID_BIAS,                         2.0},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  4},
        {ALG_PARAM_ID_MIN_LOG2_N,                   3},
    },
    // bts params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_BTS},
        {ALG_PARAM_ID_INIT_TEMP,            1.0},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,      0.05},
        {ALG_PARAM_ID_EPSILON,              0.25},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,      0.0},
    },
    // // dents params
    // {
    //     {XPR_OR_ALG_ID_TAG,                     ALG_ID_CH_DENTS},
    //     {ALG_PARAM_ID_INIT_TEMP,                1.0},
    //     {ALG_PARAM_ID_TEMP_DECAY_RATE,          0.05},
    //     {ALG_PARAM_ID_EPSILON,                  0.25},
    //     {ALG_PARAM_ID_DEFAULT_Q_VALUE,          0.0},
    //     {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       1.0},
    //     {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    100.0},
    // },
    // ch hvuct params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_HVUCT},
        {ALG_PARAM_ID_BIAS,                 4.0},
    },
    // ch pareto params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_PARETO},
        {ALG_PARAM_ID_BIAS,                 4.0},
    },
};
