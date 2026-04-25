#pragma once

#include "main_mo/configs/config_map.h"

#include "main_mo/configs/constants_algorithms.h"
#include "main_mo/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_720 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                             XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                             "720_dst_stoch_clm_scaling"},
        {XPR_PARAM_ID_ENV,                              ENV_ID_VAMPLEW_CLM_STOCH_DST_VARIABLE_CPP},
        {XPR_PARAM_ID_ENV_SIZE,                         std::vector<int>{10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100}},
        {XPR_PARAM_ID_MCTS_MODE,                        false},
        {XPR_PARAM_ID_GRAPH_SEARCH,                     true},
        {XPR_PARAM_ID_VECTOR_VISIT_COUNTS,              false},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,                 -1}, // indicates that this will be overridden for each env size
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_GLOBAL,          0.0},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_LOCAL,           0.0},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,                  true}, 
        {XPR_PARAM_ID_TERMINATION_BOUND,                15.0}, 
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,            10},
        {XPR_PARAM_ID_SEARCH_THREADS,                   16},
        {XPR_PARAM_ID_EVAL_DELTA,                       5.0}, 
        {XPR_PARAM_ID_EVAL_ROLLOUTS,                    128},
        {XPR_PARAM_ID_EVAL_THREADS,                     16},
        {XPR_PARAM_ID_CONVEX_HULL_MAX_SIZE,             15},
        {XPR_PARAM_ID_CONVEX_HULL_TOLERANCE,            1e-9},
        {XPR_PARAM_ID_USE_SOLVED_LABELLING,             true},
        {XPR_PARAM_ID_SOLVED_LABELLING_FAIL_CONFIDENCE, 0.15},
        {XPR_PARAM_ID_SOLVED_LABELLING_TOLERANCE,       0.3},

        {XPR_PARAM_ID_SM_PUSH_RADIUS,                                       10}, 
        {XPR_PARAM_ID_SM_MAX_NEIGHBOURS_TO_PUSH_TO,                         -1},
        {XPR_PARAM_ID_SM_MIN_SIMPLEX_RADIUS,                                0.01},
        {XPR_PARAM_ID_SM_SIMPLEX_SPLIT_COUNTER_THRESHOLD,                   10},
        {XPR_PARAM_ID_SM_USE_APPROX_NEAREST_VERTEX,                         false},
        {XPR_PARAM_ID_SM_EVENTUALLY_CONFORMING_SIMPLEX_MAP,                 true},
        {XPR_PARAM_ID_SM_ALWAYS_ALLOW_NON_CONFORMING_SIMPLEX_TO_SPLIT,      true},
    },
    // czt params - 0.634691
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CZT},
        {ALG_PARAM_ID_BIAS,                         1.40883},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  55},
    },
    // ch standard cheby params - 0.622105
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_STANDARD_CHEBY},
        {ALG_PARAM_ID_BIAS,                 10.0},
    },
    // ch hvuct params - 0.632238
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_HVUCT},
        {ALG_PARAM_ID_BIAS,                 20.47395},
    },
    // ch pareto params - 0.654793
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_PARETO},
        {ALG_PARAM_ID_BIAS,                 20.15138},
    },
    // ch uct params - 0.68002
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_UCT},
        {ALG_PARAM_ID_BIAS,                 4.20985},
    },
    // ch czt params - 0.626641
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CH_CZT},
        {ALG_PARAM_ID_BIAS,                         12.5507},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  49},
    },
    // ch bts params - 0.636612
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_BTS},
        {ALG_PARAM_ID_INIT_TEMP,            1e-05},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,      100},
        {ALG_PARAM_ID_EPSILON,              1.0},
    },
    // sm bts params - 0.758231
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_SM_BTS},
        {ALG_PARAM_ID_INIT_TEMP,            0.00277679},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,      0.00367017},
        {ALG_PARAM_ID_EPSILON,              0.748461},
    },
    // sm dents params - 0.770408
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_SM_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                0.0111483},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          99.9197},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       0.0132351},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    2598.62},
        {ALG_PARAM_ID_EPSILON,                  0.474472},
    },
};
