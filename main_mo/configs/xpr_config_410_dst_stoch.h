#pragma once

#include "main_mo/configs/config_map.h"

#include "main_mo/configs/constants_algorithms.h"
#include "main_mo/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_410 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                             XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                             "410_dst_stoch"},
        {XPR_PARAM_ID_ENV,                              ENV_ID_VAMPLEW_STOCH_DST_10_CPP},
        {XPR_PARAM_ID_ENV_SIZE,                         NO_ENV_SIZE},
        {XPR_PARAM_ID_MCTS_MODE,                        false},
        {XPR_PARAM_ID_GRAPH_SEARCH,                     true},
        {XPR_PARAM_ID_VECTOR_VISIT_COUNTS,              false},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,                 26*2}, 
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_GLOBAL,          0.0},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_LOCAL,           0.0},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,                  true}, 
        {XPR_PARAM_ID_TERMINATION_BOUND,                15.0}, 
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,            10},
        {XPR_PARAM_ID_SEARCH_THREADS,                   16},
        {XPR_PARAM_ID_EVAL_DELTA,                       0.05}, 
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
    // chvi
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CHVI},
    },
    // // chvi ordered
    // {
    //     {XPR_OR_ALG_ID_TAG,                 ALG_ID_CHVI_ORDERED},
    // },
    // // chvi reversed
    // {
    //     {XPR_OR_ALG_ID_TAG,                 ALG_ID_CHVI_REVERSED},
    // },
    // czt params - 0.584191
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CZT},
        {ALG_PARAM_ID_BIAS,                         2.03847},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  150},
    },
    // ch standard cheby params - 0.4073
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_STANDARD_CHEBY},
        {ALG_PARAM_ID_BIAS,                 1.0972}, //1.0972e-05},
    },
    // ch hvuct params - 0.434612
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_HVUCT},
        {ALG_PARAM_ID_BIAS,                 15.9883},
    },
    // ch pareto params - 0.405224
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_PARETO},
        {ALG_PARAM_ID_BIAS,                 0.365202}, //0.00365202},
    },
    // ch uct params - 0.444231
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_UCT},
        {ALG_PARAM_ID_BIAS,                 4.20985}, //0.000420985},
    },
    // ch czt params - 0.506634
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CH_CZT},
        {ALG_PARAM_ID_BIAS,                         5.81366},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  40},
    },
    // ch bts params - 0.429332
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_BTS},
        {ALG_PARAM_ID_INIT_TEMP,            1.00157e-05},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,      4.51053},
        {ALG_PARAM_ID_EPSILON,              0.999874},
    },
    // sm bts params - 0.690184
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_SM_BTS},
        {ALG_PARAM_ID_INIT_TEMP,            1e-05},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,      0.001},
        {ALG_PARAM_ID_EPSILON,              0.85919},
    },
    // sm dents params - 0.726752
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_SM_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                0.000388414},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          0.0940159},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       0.0182551},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    1598.62},
        {ALG_PARAM_ID_EPSILON,                  0.374472},
    },
};
