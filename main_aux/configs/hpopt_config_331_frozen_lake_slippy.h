#pragma once

#include "main_aux/configs/hpopt_config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"
#include "main_aux/configs/constants_hpopt.h"

// Notes:
// Epsilon generally gets tuned to 1.0, it can basically solve 8x8 problem by itself
// But set it to 0.0 in tuning to tune the rest of the parameters
// For when rest of parameters are relevant on larger problems

static const std::vector<HpoptConfigMap> HPOPT_CONFIG_331 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                             HPOPT_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                             "331_hpopt_frozen_lake_slippy"},
        {XPR_PARAM_ID_ENV,                              ENV_ID_SLIPPY_FROZEN_LAKE_S_4x4},
        {XPR_PARAM_ID_MCTS_MODE,                        false},
        {XPR_PARAM_ID_GRAPH_SEARCH,                     true},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,                 100},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_GLOBAL,          0.0},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_LOCAL,           0.0},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,                  false},
        {XPR_PARAM_ID_TERMINATION_BOUND,                100000},
        // {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,            2},
        {XPR_PARAM_ID_SEARCH_THREADS,                   16},
        {XPR_PARAM_ID_EVAL_DELTA,                       100000},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,                    1024},
        {XPR_PARAM_ID_EVAL_THREADS,                     16},
        {HPOPT_PARAM_ID_MIN_REPEATS,                    10},
        {HPOPT_PARAM_ID_ESTIMATE_CONFIDENCE_THRESHOLD,  0.035},
        {HPOPT_PARAM_ID_BAYESOPT_TOTAL_SAMPLES,         75},
        {HPOPT_PARAM_ID_BAYESOPT_INIT_RAND_SAMPLES,     10},
        {HPOPT_PARAM_ID_BAYESOPT_RELEARN_FREQ,          20},
        {HPOPT_PARAM_ID_BAYESOPT_USE_GPML,              1},
    },
    // uct params
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS, std::make_pair(1.0e-5,  1.0e2)},
        {ALG_PARAM_ID_HEURISTIC_VALUE, std::make_pair(1.0,  1.0)},
    },
    // max uct params
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_MAX_UCT},
        {ALG_PARAM_ID_BIAS, std::make_pair(1.0e-5,  1.0e2)},
        {ALG_PARAM_ID_HEURISTIC_VALUE, std::make_pair(1.0,  1.0)},
    },
    // hmcts params
    {
        {XPR_OR_ALG_ID_TAG,         ALG_ID_HMCTS},
        {ALG_PARAM_ID_BIAS,         std::make_pair(1.0e-5,  1.0e2)},
        {ALG_PARAM_ID_UCT_BUDGET,   std::make_pair(1.0,     1.0e5)},
        {ALG_PARAM_ID_HEURISTIC_VALUE, std::make_pair(1.0,  1.0)},
    },
};
