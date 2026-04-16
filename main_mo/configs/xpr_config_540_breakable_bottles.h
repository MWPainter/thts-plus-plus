#pragma once

#include "main_mo/configs/config_map.h"

#include "main_mo/configs/constants_algorithms.h"
#include "main_mo/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_540 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                             XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                             "540_breakable_bottles"},
        {XPR_PARAM_ID_ENV,                              ENV_ID_BREAKABLE_BOTTLES},
        {XPR_PARAM_ID_ENV_SIZE,                         NO_ENV_SIZE},
        {XPR_PARAM_ID_MCTS_MODE,                        false},
        {XPR_PARAM_ID_GRAPH_SEARCH,                     true},
        {XPR_PARAM_ID_VECTOR_VISIT_COUNTS,              false},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,                 100},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_GLOBAL,          0.0},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_LOCAL,           0.0},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,                  true}, 
        {XPR_PARAM_ID_TERMINATION_BOUND,                60.0}, // 90 sec
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,            10},
        {XPR_PARAM_ID_SEARCH_THREADS,                   16},
        {XPR_PARAM_ID_EVAL_DELTA,                       0.5}, // log every 1 second
        {XPR_PARAM_ID_EVAL_ROLLOUTS,                    1024},
        {XPR_PARAM_ID_EVAL_THREADS,                     16},
        {XPR_PARAM_ID_CONVEX_HULL_MAX_SIZE,             15},
        {XPR_PARAM_ID_CONVEX_HULL_TOLERANCE,            1e-9},
        {XPR_PARAM_ID_USE_SOLVED_LABELLING,             true},
        {XPR_PARAM_ID_SOLVED_LABELLING_FAIL_CONFIDENCE, 0.05},
        {XPR_PARAM_ID_SOLVED_LABELLING_TOLERANCE,       0.1},

        {XPR_PARAM_ID_SM_PUSH_RADIUS,                                       10}, 
        {XPR_PARAM_ID_SM_MAX_NEIGHBOURS_TO_PUSH_TO,                         -1},
        {XPR_PARAM_ID_SM_MIN_SIMPLEX_RADIUS,                                0.01},
        {XPR_PARAM_ID_SM_SIMPLEX_SPLIT_COUNTER_THRESHOLD,                   10},
        {XPR_PARAM_ID_SM_USE_APPROX_NEAREST_VERTEX,                         false},
        {XPR_PARAM_ID_SM_EVENTUALLY_CONFORMING_SIMPLEX_MAP,                 true},
        {XPR_PARAM_ID_SM_ALWAYS_ALLOW_NON_CONFORMING_SIMPLEX_TO_SPLIT,      true},
    },
    // czt params
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CZT},
        {ALG_PARAM_ID_BIAS,                         1.93943},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  10},
    },
    // ch standard cheby params - 0.74242
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_STANDARD_CHEBY},
        {ALG_PARAM_ID_BIAS,                 0.617918},
    },
    // ch hvuct params - 0.705814
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_HVUCT},
        {ALG_PARAM_ID_BIAS,                 1000.0},
    },
    // ch pareto params - 0.731513
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_PARETO},
        {ALG_PARAM_ID_BIAS,                 1.0},
    },
    // ch uct params - 0.763165
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_UCT},
        {ALG_PARAM_ID_BIAS,                 1.0},
    },
    // ch czt params - 0.668835
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CH_CZT},
        {ALG_PARAM_ID_BIAS,                         511.909},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  10},
    },
    // ch bts params - 0.758271
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_BTS},
        {ALG_PARAM_ID_INIT_TEMP,            0.000175013},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,      0.00982467},
        {ALG_PARAM_ID_EPSILON,              1},
    },
    // sm bts params - 0.683374
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_SM_BTS},
        {ALG_PARAM_ID_INIT_TEMP,            0.0165908},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,      14.1084},
        {ALG_PARAM_ID_EPSILON,              0.5},
    },
    // sm dents params
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_SM_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                0.0165908},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          14.1084},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       0.05},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    5000},
        {ALG_PARAM_ID_EPSILON,                  0.5},
    },
};
