#pragma once

#include "main_mo/configs/hpopt_config_map.h"

#include "main_mo/configs/constants_algorithms.h"
#include "main_mo/configs/constants_experiments.h"
#include "main_mo/configs/constants_hpopt.h"

static const std::vector<HpoptConfigMap> HPOPT_CONFIG_001 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                             HPOPT_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                             "001_debug_hpopt"},
        {XPR_PARAM_ID_ENV,                              ENV_ID_PY_DEBUG_2},
        {XPR_PARAM_ID_MCTS_MODE,                        true},
        {XPR_PARAM_ID_GRAPH_SEARCH,                     true},
        {XPR_PARAM_ID_VECTOR_VISIT_COUNTS,              false},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,                 10000},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,                  true},
        {XPR_PARAM_ID_TERMINATION_BOUND,                0.25},
        // {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,            2},
        {XPR_PARAM_ID_SEARCH_THREADS,                   16},
        {XPR_PARAM_ID_EVAL_DELTA,                       0.25},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,                    128},
        {XPR_PARAM_ID_EVAL_THREADS,                     16},
        {XPR_PARAM_ID_CONVEX_HULL_MAX_SIZE,             -1},
        {XPR_PARAM_ID_CONVEX_HULL_TOLERANCE,            1e-9},
        {XPR_PARAM_ID_USE_SOLVED_LABELLING,              true},
        {XPR_PARAM_ID_SOLVED_LABELLING_FAIL_CONFIDENCE, 0.05},
        {XPR_PARAM_ID_SOLVED_LABELLING_TOLERANCE,         0.1},

        {XPR_PARAM_ID_SM_PUSH_RADIUS,                                       10}, 
        {XPR_PARAM_ID_SM_MAX_NEIGHBOURS_TO_PUSH_TO,                         -1},
        {XPR_PARAM_ID_SM_MIN_SIMPLEX_RADIUS,                                0.01},
        {XPR_PARAM_ID_SM_SIMPLEX_SPLIT_COUNTER_THRESHOLD,                   10},
        {XPR_PARAM_ID_SM_USE_APPROX_NEAREST_VERTEX,                         false},
        {XPR_PARAM_ID_SM_EVENTUALLY_CONFORMING_SIMPLEX_MAP,                 true},
        {XPR_PARAM_ID_SM_ALWAYS_ALLOW_NON_CONFORMING_SIMPLEX_TO_SPLIT,      true},

        {HPOPT_PARAM_ID_MIN_REPEATS,                    10},
        {HPOPT_PARAM_ID_ESTIMATE_CONFIDENCE_THRESHOLD,  1.0},
        {HPOPT_PARAM_ID_BAYESOPT_TOTAL_SAMPLES,         30},
        {HPOPT_PARAM_ID_BAYESOPT_INIT_RAND_SAMPLES,     10},
        {HPOPT_PARAM_ID_BAYESOPT_RELEARN_FREQ,          20},
    },
    // czt params
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CZT},
        {ALG_PARAM_ID_BIAS,                         std::make_pair(1.0e-4,  1.0e4)},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  std::make_pair(1.0,     1024.0)},
    },
    // czt doubling params
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CZT_DOUBLING},
        {ALG_PARAM_ID_BIAS,                         std::make_pair(1.0e-4,  1.0e4)},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  std::make_pair(1.0,     1024.0)},
        {ALG_PARAM_ID_MIN_LOG2_N,                   std::make_pair(0.0,     10.0)},
    },
    // ch uct params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_UCT},
        {ALG_PARAM_ID_BIAS,                 std::make_pair(1.0e-4,   1.0e4)},
    },
    // ch czt params
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CH_CZT},
        {ALG_PARAM_ID_BIAS,                         std::make_pair(1.0e-4,  1.0e4)},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  std::make_pair(1.0,     1024.0)},
    },
    // ch czt doubling params
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CH_CZT_DOUBLING},
        {ALG_PARAM_ID_BIAS,                         std::make_pair(1.0e-4,  1.0e4)},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  std::make_pair(1.0,     1024.0)},
        {ALG_PARAM_ID_MIN_LOG2_N,                   std::make_pair(0.0,     10.0)},
    },
    // bts params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_BTS},
        {ALG_PARAM_ID_INIT_TEMP,            std::make_pair(1.0e-4,  1.0e4)},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,      std::make_pair(1.0e-4,  1.0e4)},
        {ALG_PARAM_ID_EPSILON,              std::make_pair(1.0e-6,  1.0e0)},
    },
    // // dents params
    // {
    //     {XPR_OR_ALG_ID_TAG,                     ALG_ID_CH_DENTS},
    //     {ALG_PARAM_ID_INIT_TEMP,                std::make_pair(1.0e-4,  1.0e4)},
    //     {ALG_PARAM_ID_TEMP_DECAY_RATE,          std::make_pair(1.0e-4,  1.0e4)},
    //     {ALG_PARAM_ID_EPSILON,                  std::make_pair(1.0e-6,  1.0e0)},
    //     {ALG_PARAM_ID_DEFAULT_Q_VALUE,          std::make_pair(-100.0,  0.0)},
    //     {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       std::make_pair(1.0e-4,  1.0e4)},
    //     {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    std::make_pair(1.0e0,   1.0e4)},
    // },
    // ch hvuct params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_HVUCT},
        {ALG_PARAM_ID_BIAS,                 std::make_pair(1.0e-4,   1.0e4)},
    },
    // ch pareto params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_PARETO},
        {ALG_PARAM_ID_BIAS,                 std::make_pair(1.0e-4,   1.0e4)},
    },
};
