#pragma once

#include "main_mo/configs/config_map.h"

#include "main_mo/configs/constants_algorithms.h"
#include "main_mo/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_460 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                             XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                             "460_dst_improved_stoch_clm"},
        {XPR_PARAM_ID_ENV,                              ENV_ID_IMPROVED_CLM_STOCH_DST},
        {XPR_PARAM_ID_ENV_SIZE,                         NO_ENV_SIZE},
        {XPR_PARAM_ID_MCTS_MODE,                        false},
        {XPR_PARAM_ID_GRAPH_SEARCH,                     true},
        {XPR_PARAM_ID_VECTOR_VISIT_COUNTS,              false},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,                 26*2}, 
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_GLOBAL,          0.0},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_LOCAL,           0.0},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,                  true}, 
        {XPR_PARAM_ID_TERMINATION_BOUND,                60.0}, 
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,            10},
        {XPR_PARAM_ID_SEARCH_THREADS,                   16},
        {XPR_PARAM_ID_EVAL_DELTA,                       0.5},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,                    256},
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
    // chvi ordered
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CHVI_ORDERED},
    },
    // chvi reversed
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CHVI_REVERSED},
    },
    // czt params - 0.664511
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CZT},
        {ALG_PARAM_ID_BIAS,                         1.40883},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  55},
    },
    // ch standard cheby params - 0.606425
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_STANDARD_CHEBY},
        {ALG_PARAM_ID_BIAS,                 11.4923},
    },
    // ch hvuct params - 0.602044
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_HVUCT},
        {ALG_PARAM_ID_BIAS,                 20.4423},
    },
    // ch pareto params - 0.607654
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_PARETO},
        {ALG_PARAM_ID_BIAS,                 0.215138},
    },
    // ch uct params - 0.60741
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_UCT},
        {ALG_PARAM_ID_BIAS,                 8.3396},
    },
    // ch czt params - 0.573512
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CH_CZT},
        {ALG_PARAM_ID_BIAS,                         43.4261},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  980},
    },
    // ch bts params - 0.625076
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_BTS},
        {ALG_PARAM_ID_INIT_TEMP,            0.00135245},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,      99.9823},
        {ALG_PARAM_ID_EPSILON,              1.0},
    },
    // sm bts params - 0.611868
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_SM_BTS},
        {ALG_PARAM_ID_INIT_TEMP,            0.00277679},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,      0.00367017},
        {ALG_PARAM_ID_EPSILON,              0.748461},
    },
    // sm dents params - 0.604783
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_SM_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                0.00668512},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          0.00876359},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       19.9579},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    1673.33},
        {ALG_PARAM_ID_EPSILON,                  0.474472},
    },
};
