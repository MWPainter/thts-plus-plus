#pragma once

#include "main_mo/configs/config_map.h"

#include "main_mo/configs/constants_algorithms.h"
#include "main_mo/configs/constants_experiments.h"

static const std::vector<ConfigMap> CONFIG_003 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                             XPR_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                             "003_debug"},
        {XPR_PARAM_ID_ENV,                              ENV_ID_VAMPLEW_DST_10},
        {XPR_PARAM_ID_MCTS_MODE,                        true},
        {XPR_PARAM_ID_GRAPH_SEARCH,                     true},
        {XPR_PARAM_ID_VECTOR_VISIT_COUNTS,              false},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,                 26*2},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,                  true},
        {XPR_PARAM_ID_TERMINATION_BOUND,                30.0},
        {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,            25},
        {XPR_PARAM_ID_SEARCH_THREADS,                   16},
        {XPR_PARAM_ID_EVAL_DELTA,                       1.0},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,                    1024},
        {XPR_PARAM_ID_EVAL_THREADS,                     16},
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
