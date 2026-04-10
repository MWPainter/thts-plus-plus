#pragma once

#include "main_mo/configs/hpopt_config_map.h"

#include "main_mo/configs/constants_algorithms.h"
#include "main_mo/configs/constants_experiments.h"

static const std::vector<HpoptConfigMap> HPOPT_CONFIG_140 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                             HPOPT_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                             "140_hpopt_dst_improved"},
        {XPR_PARAM_ID_ENV,                              ENV_ID_IMPROVED_DST},
        // {XPR_PARAM_ID_ENV_SIZE,                         NO_ENV_SIZE},
        {XPR_PARAM_ID_MCTS_MODE,                        false},
        {XPR_PARAM_ID_GRAPH_SEARCH,                     true},
        {XPR_PARAM_ID_VECTOR_VISIT_COUNTS,              false},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,                 20*2},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_GLOBAL,          0.0},
        {XPR_PARAM_ID_HEURISTIC_WEIGHT_LOCAL,           0.0}, 
        {XPR_PARAM_ID_RUNTIME_BOUNDED,                  true}, 
        {XPR_PARAM_ID_TERMINATION_BOUND,                1.0}, // give each alg 1 second to run (going to do a lot of repeats) - if get bad results can try longer
        // {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,            2},
        {XPR_PARAM_ID_SEARCH_THREADS,                   16},
        {XPR_PARAM_ID_EVAL_DELTA,                       1.0}, 
        {XPR_PARAM_ID_EVAL_ROLLOUTS,                    1024},
        {XPR_PARAM_ID_EVAL_THREADS,                     16},
        {XPR_PARAM_ID_CONVEX_HULL_MAX_SIZE,             25},
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

        {HPOPT_PARAM_ID_MIN_REPEATS,                    10},
        {HPOPT_PARAM_ID_ESTIMATE_CONFIDENCE_THRESHOLD,  0.025},
        {HPOPT_PARAM_ID_BAYESOPT_TOTAL_SAMPLES,         50}, // initially try 50 samples, do more if get bad results on some envs?
        {HPOPT_PARAM_ID_BAYESOPT_INIT_RAND_SAMPLES,     10},
        {HPOPT_PARAM_ID_BAYESOPT_RELEARN_FREQ,          20},
        {HPOPT_PARAM_ID_BAYESOPT_USE_GPML,              0},
    },
    // czt params
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CZT},
        {ALG_PARAM_ID_BIAS,                         std::make_pair(1.0e-5,  1.0e5)},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  std::make_pair(1.0,     1024.0)},
    },
    // czt doubling params
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CZT_DOUBLING},
        {ALG_PARAM_ID_BIAS,                         std::make_pair(1.0e-5,  1.0e5)},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  std::make_pair(1.0,     1024.0)},
        {ALG_PARAM_ID_MIN_LOG2_N,                   std::make_pair(0.0,     10.0)},
    },
    // ch cheby params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_CHEBY},
        {ALG_PARAM_ID_BIAS,                 std::make_pair(1.0e-5,   1.0e5)},
    },
    // ch standard cheby params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_STANDARD_CHEBY},
        {ALG_PARAM_ID_BIAS,                 std::make_pair(1.0e-5,   1.0e5)},
    },
    // ch hvuct params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_HVUCT},
        {ALG_PARAM_ID_BIAS,                 std::make_pair(1.0e-5,   1.0e5)},
    },
    // ch pareto params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_PARETO},
        {ALG_PARAM_ID_BIAS,                 std::make_pair(1.0e-5,   1.0e5)},
    },
    // ch uct params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_UCT},
        {ALG_PARAM_ID_BIAS,                 std::make_pair(1.0e-5,   1.0e5)},
    },
    // ch czt params
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CH_CZT},
        {ALG_PARAM_ID_BIAS,                         std::make_pair(1.0e-5,   1.0e5)},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  std::make_pair(1.0,     1024.0)},
    },
    // ch czt doubling params
    {
        {XPR_OR_ALG_ID_TAG,                         ALG_ID_CH_CZT_DOUBLING},
        {ALG_PARAM_ID_BIAS,                         std::make_pair(1.0e-5,   1.0e5)},
        {ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH,  std::make_pair(1.0,     1024.0)},
        {ALG_PARAM_ID_MIN_LOG2_N,                   std::make_pair(0.0,     10.0)},
    },
    // ch bts params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_CH_BTS},
        {ALG_PARAM_ID_INIT_TEMP,            std::make_pair(1.0e-5,  1.0e2)},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,      std::make_pair(1.0e-3,  1.0e2)},
        {ALG_PARAM_ID_EPSILON,              std::make_pair(1.0e-6,1.0)}, 
    },
    // sm bts params
    {
        {XPR_OR_ALG_ID_TAG,                 ALG_ID_SM_BTS},
        {ALG_PARAM_ID_INIT_TEMP,            std::make_pair(1.0e-5,  1.0e2)},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,      std::make_pair(1.0e-3,  1.0e2)},
        {ALG_PARAM_ID_EPSILON,              std::make_pair(1.0e-6,1.0)}, 
    },
    // sm dents params
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_SM_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                std::make_pair(1.0e-5,  1.0e2)},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          std::make_pair(1.0e-3,  1.0e2)},
        {ALG_PARAM_ID_EPSILON,                  std::make_pair(1.0e-6,1.0)}, 
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       std::make_pair(1.0e-5,  1.0e2)},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    std::make_pair(1.0e3,  1.0e5)},
    },
};
