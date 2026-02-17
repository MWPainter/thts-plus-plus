#pragma once

#include "main_mo/configs/config_map.h"

#include "main_mo/configs/constants_algorithms.h"
#include "main_mo/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_040 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                             XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                             "040_debug_test_cheby"},
        {XPR_PARAM_ID_ENV,                              ENV_ID_VAMPLEW_DST_10_CPP},
        {XPR_PARAM_ID_ENV_SIZE,                         NO_ENV_SIZE},
        {XPR_PARAM_ID_MCTS_MODE,                        true},
        {XPR_PARAM_ID_GRAPH_SEARCH,                     true},
        {XPR_PARAM_ID_VECTOR_VISIT_COUNTS,              false},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,                 26*2},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,                  true}, 
        {XPR_PARAM_ID_TERMINATION_BOUND,                10.0}, // 90 sec
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,            1},
        {XPR_PARAM_ID_SEARCH_THREADS,                   16},
        {XPR_PARAM_ID_EVAL_DELTA,                       1.0}, // log every 1 second
        {XPR_PARAM_ID_EVAL_ROLLOUTS,                    1},
        {XPR_PARAM_ID_EVAL_THREADS,                     1},
        {XPR_PARAM_ID_CONVEX_HULL_MAX_SIZE,             -1},
        {XPR_PARAM_ID_CONVEX_HULL_TOLERANCE,            1e-9},
        {XPR_PARAM_ID_USE_SOLVED_LABELLING,              false},
        {XPR_PARAM_ID_SOLVED_LABELLING_FAIL_CONFIDENCE, 0.05},
        {XPR_PARAM_ID_SOLVED_LABELLING_TOLERANCE,         0.1},
    },
    // cheby params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_CHEBY},
        {ALG_PARAM_ID_BIAS,                 10.0},
    },
};
