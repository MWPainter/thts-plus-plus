#pragma once

#include "main_mo/configs/config_map.h"

#include "main_mo/configs/constants_algorithms.h"
#include "main_mo/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_072 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                             XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                             "072_debug_final_test_all_algs_det_mcts_local_heuristic"},
        {XPR_PARAM_ID_ENV,                              ENV_ID_VAMPLEW_DST_10_CPP},
        {XPR_PARAM_ID_ENV_SIZE,                         NO_ENV_SIZE},
        {XPR_PARAM_ID_MCTS_MODE,                        true},
        {XPR_PARAM_ID_GRAPH_SEARCH,                     true},
        {XPR_PARAM_ID_VECTOR_VISIT_COUNTS,              false},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,                 26*2},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_GLOBAL,          1.0},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_LOCAL,           1.0}, 
        {XPR_PARAM_ID_RUNTIME_BOUNDED,                  true}, 
        {XPR_PARAM_ID_TERMINATION_BOUND,                10.0}, // 90 sec
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,            2},
        {XPR_PARAM_ID_SEARCH_THREADS,                   16},
        {XPR_PARAM_ID_EVAL_DELTA,                       1.0}, // log every 1 second
        {XPR_PARAM_ID_EVAL_ROLLOUTS,                    256},
        {XPR_PARAM_ID_EVAL_THREADS,                     16},
        {XPR_PARAM_ID_CONVEX_HULL_MAX_SIZE,             25},
        {XPR_PARAM_ID_CONVEX_HULL_TOLERANCE,            1e-9},
        {XPR_PARAM_ID_USE_SOLVED_LABELLING,              true},
        {XPR_PARAM_ID_SOLVED_LABELLING_FAIL_CONFIDENCE, 0.05},
        {XPR_PARAM_ID_SOLVED_LABELLING_TOLERANCE,         0.1},

        {XPR_PARAM_ID_SM_PUSH_RADIUS,                                       10}, 
        {XPR_PARAM_ID_SM_MAX_NEIGHBOURS_TO_PUSH_TO,                         -1},
        {XPR_PARAM_ID_SM_MIN_SIMPLEX_RADIUS,                                0.01},
        {XPR_PARAM_ID_SM_SIMPLEX_SPLIT_COUNTER_THRESHOLD,                   10},
        {XPR_PARAM_ID_SM_USE_APPROX_NEAREST_VERTEX,                         false},
        {XPR_PARAM_ID_SM_EVENTUALLY_CONFORMING_SIMPLEX_MAP,                 true},
        {XPR_PARAM_ID_SM_ALWAYS_ALLOW_NON_CONFORMING_SIMPLEX_TO_SPLIT,      true},
    },
    // chvi
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CHVI},
    },
    // czt params
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CZT},
        {ALG_PARAM_ID_BIAS,                         4.0},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  4},
    },
    // czt doubling params
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CZT_DOUBLING},
        {ALG_PARAM_ID_BIAS,                         4.0},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  4},
        {ALG_PARAM_ID_MIN_LOG2_N,                   3},
    },
    // ch cheby params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_CHEBY},
        {ALG_PARAM_ID_BIAS,                 10.0},
    },
    // ch standard cheby params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_STANDARD_CHEBY},
        {ALG_PARAM_ID_BIAS,                 10.0},
    },
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
    // ch uct params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_UCT},
        {ALG_PARAM_ID_BIAS,                 4.0},
    },
    // ch czt params
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CH_CZT},
        {ALG_PARAM_ID_BIAS,                         4.0},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  4},
    },
    // ch czt doubling params
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CH_CZT_DOUBLING},
        {ALG_PARAM_ID_BIAS,                         4.0},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  4},
        {ALG_PARAM_ID_MIN_LOG2_N,                   3},
    },
    // ch bts params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_BTS},
        {ALG_PARAM_ID_INIT_TEMP,            1.0},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,      0.05},
        {ALG_PARAM_ID_EPSILON,              1.0},
    },
    // sm bts params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_SM_BTS},
        {ALG_PARAM_ID_INIT_TEMP,            1.0},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,      0.05},
        {ALG_PARAM_ID_EPSILON,              1.0},
    },
    // sm dents params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_SM_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,            1.0},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,      0.05},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,   1.0},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT, 1000.0},
        {ALG_PARAM_ID_EPSILON,              1.0},
    },
};
