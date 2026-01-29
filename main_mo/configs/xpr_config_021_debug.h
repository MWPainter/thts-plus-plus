#pragma once

#include "main_mo/configs/config_map.h"

#include "main_mo/configs/constants_algorithms.h"
#include "main_mo/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_021 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                             XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                             "021_debug"},
        {XPR_PARAM_ID_ENV,                              ENV_ID_VAMPLEW_STOCH_DST_VARIABLE_CPP},
        {XPR_PARAM_ID_ENV_SIZE,                         std::vector<int>{10, 15}},
        {XPR_PARAM_ID_MCTS_MODE,                        true},
        {XPR_PARAM_ID_GRAPH_SEARCH,                     true},
        {XPR_PARAM_ID_VECTOR_VISIT_COUNTS,              false},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,                 -1}, // indicates that this will be overridden for each env size
        {XPR_PARAM_ID_RUNTIME_BOUNDED,                  true}, 
        {XPR_PARAM_ID_TERMINATION_BOUND,                10.0}, // 90 sec
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,            5},
        {XPR_PARAM_ID_SEARCH_THREADS,                   16},
        {XPR_PARAM_ID_EVAL_DELTA,                       1.0}, // log every 1 second
        {XPR_PARAM_ID_EVAL_ROLLOUTS,                    1},
        {XPR_PARAM_ID_EVAL_THREADS,                     1},
        {XPR_PARAM_ID_CONVEX_HULL_MAX_SIZE,             25},
        {XPR_PARAM_ID_CONVEX_HULL_TOLERANCE,            1e-9},
    },
    // chvi
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CHVI},
    },
    // bts params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_BTS},
        {ALG_PARAM_ID_INIT_TEMP,            1.0},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,      0.05},
        {ALG_PARAM_ID_EPSILON,              1.0},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,      0.0},
    },
};
