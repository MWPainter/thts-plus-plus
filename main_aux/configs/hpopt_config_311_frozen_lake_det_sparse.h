#pragma once

#include "main_aux/configs/hpopt_config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"
#include "main_aux/configs/constants_hpopt.h"

static const std::vector<HpoptConfigMap> HPOPT_CONFIG_311 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                             HPOPT_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                             "311_hpopt_frozen_lake_det_sparse"},
        {XPR_PARAM_ID_ENV,                              ENV_ID_FROZEN_LAKE_S_8x8},
        {XPR_PARAM_ID_MCTS_MODE,                        false},
        {XPR_PARAM_ID_GRAPH_SEARCH,                     false},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,                 100},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,                  false},
        {XPR_PARAM_ID_TERMINATION_BOUND,                10000},
        // {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,            2},
        {XPR_PARAM_ID_SEARCH_THREADS,                   16},
        {XPR_PARAM_ID_EVAL_DELTA,                       10000},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,                    1},
        {XPR_PARAM_ID_EVAL_THREADS,                     1},
        {HPOPT_PARAM_ID_MIN_REPEATS,                    10},
        {HPOPT_PARAM_ID_ESTIMATE_CONFIDENCE_THRESHOLD,  0.015},
        {HPOPT_PARAM_ID_BAYESOPT_TOTAL_SAMPLES,         200},
        {HPOPT_PARAM_ID_BAYESOPT_INIT_RAND_SAMPLES,     10},
        {HPOPT_PARAM_ID_BAYESOPT_RELEARN_FREQ,          20},
    },
    // uct params
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS, std::make_pair(1.0e-4,  1.0e4)},
    },
    // max uct params
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_MAX_UCT},
        {ALG_PARAM_ID_BIAS, std::make_pair(1.0e-4,  1.0e4)},
    },
    // hmcts params
    {
        {XPR_OR_ALG_ID_TAG,         ALG_ID_HMCTS},
        {ALG_PARAM_ID_BIAS,         std::make_pair(1.0e-4,  1.0e4)},
        {ALG_PARAM_ID_UCT_BUDGET,   std::make_pair(1.0,     5000.0)},
    },
};
