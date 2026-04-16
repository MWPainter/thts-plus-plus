#pragma once

#include "main_mo/configs/config_map.h"

#include "main_mo/configs/constants_algorithms.h"
#include "main_mo/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_560 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                             XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                             "560_four_room_timed"},
        {XPR_PARAM_ID_ENV,                              ENV_ID_FOUR_ROOM_TIMED},
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
    // czt params - 0.302133
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CZT},
        {ALG_PARAM_ID_BIAS,                         1.0},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  15},
    },
    // ch standard cheby params - 0.160655
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_STANDARD_CHEBY},
        {ALG_PARAM_ID_BIAS,                 40.7322},
    },
    // ch hvuct params - 0.154999
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_HVUCT},
        {ALG_PARAM_ID_BIAS,                 1000.0},
    },
    // ch pareto params - 0.16528
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_PARETO},
        {ALG_PARAM_ID_BIAS,                 1000.0},
    },
    // ch uct params - 0.164974
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_UCT},
        {ALG_PARAM_ID_BIAS,                 1000.0},
    },
    // ch czt params - 0.152228
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CH_CZT},
        {ALG_PARAM_ID_BIAS,                         1000.0},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  32},
    },
    // ch bts params - 0.168428
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_BTS},
        {ALG_PARAM_ID_INIT_TEMP,            0.186598},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,      0.00100105},
        {ALG_PARAM_ID_EPSILON,              0.150888},
    },
    // sm bts params - 0.153437
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_SM_BTS},
        {ALG_PARAM_ID_INIT_TEMP,            0.0829886},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,      0.001},
        {ALG_PARAM_ID_EPSILON,              0.074457},
    },
    // sm dents params - 0.147143
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_SM_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                0.0829886},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          0.001},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       0.0100149},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    1000},
        {ALG_PARAM_ID_EPSILON,                  0.074457},
    },
};
