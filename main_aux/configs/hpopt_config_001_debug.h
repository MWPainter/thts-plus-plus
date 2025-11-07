#pragma once

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"
#include "main_aux/configs/constants_hpopt.h"

static const std::vector<HpoptConfigMap> HPOPT_CONFIG_001 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                             HPOPT_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                             "001_debug_hpopt"},
        {XPR_PARAM_ID_ENV,                              ENV_ID_D_CHAIN_10},
        {XPR_PARAM_ID_MCTS_MODE,                        true},
        {XPR_PARAM_ID_GRAPH_SEARCH,                     false},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,                 10000},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,                  false},
        {XPR_PARAM_ID_TERMINATION_BOUND,                100},
        // {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,            2},
        {XPR_PARAM_ID_SEARCH_THREADS,                   2},
        {XPR_PARAM_ID_EVAL_DELTA,                       25},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,                    25},
        {XPR_PARAM_ID_EVAL_THREADS,                     2},
        {HPOPT_PARAM_ID_MIN_REPEATS,                    5},
        {HPOPT_PARAM_ID_ESTIMATE_CONFIDENCE_THRESHOLD,  1.0},
        {HPOPT_PARAM_ID_BAYESOPT_TOTAL_SAMPLES,         200},
        {HPOPT_PARAM_ID_BAYESOPT_INIT_RAND_SAMPLES,     10},
        {HPOPT_PARAM_ID_BAYESOPT_RELEARN_FREQ,          10},
    },
    // uct params
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS, std::make_pair(0.001,   1000)},
    },
    // max uct params
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_MAX_UCT},
        {ALG_PARAM_ID_BIAS, std::make_pair(0.001,   1000)},
    },
    // hmcts params
    {
        {XPR_OR_ALG_ID_TAG,         ALG_ID_HMCTS},
        {ALG_PARAM_ID_BIAS,         std::make_pair(1.0e-3,  1.0e4)},
        {ALG_PARAM_ID_UCT_BUDGET,   std::make_pair(1.0,     100.0)},
    },
    // ments params
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        std::make_pair(1.0e-4,  1.0e4)},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  std::make_pair(1.0e-4,  1.0e4)},
        {ALG_PARAM_ID_EPSILON,          std::make_pair(1.0e-6,  1.0e0)},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  std::make_pair(-100.0,  0.0)},
    },
    // rents params
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_RENTS},
        {ALG_PARAM_ID_INIT_TEMP,        std::make_pair(1.0e-4,  1.0e4)},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  std::make_pair(1.0e-4,  1.0e4)},
        {ALG_PARAM_ID_EPSILON,          std::make_pair(1.0e-6,  1.0e0)},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  std::make_pair(-100.0,  0.0)},
    },
    // tents params
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_TENTS},
        {ALG_PARAM_ID_INIT_TEMP,        std::make_pair(1.0e-4,  1.0e4)},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  std::make_pair(1.0e-4,  1.0e4)},
        {ALG_PARAM_ID_EPSILON,          std::make_pair(1.0e-6,  1.0e0)},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  std::make_pair(-100.0,  0.0)},
    },
    // bts params
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_BTS},
        {ALG_PARAM_ID_INIT_TEMP,        std::make_pair(1.0e-4,  1.0e4)},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  std::make_pair(1.0e-4,  1.0e4)},
        {ALG_PARAM_ID_EPSILON,          std::make_pair(1.0e-6,  1.0e0)},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  std::make_pair(-100.0,  0.0)},
    },
    // dents params
    {
        {XPR_OR_ALG_ID_TAG,                     ALG_ID_DENTS},
        {ALG_PARAM_ID_INIT_TEMP,                std::make_pair(1.0e-4,  1.0e4)},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,          std::make_pair(1.0e-4,  1.0e4)},
        {ALG_PARAM_ID_EPSILON,                  std::make_pair(1.0e-6,  1.0e0)},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,          std::make_pair(-100.0,  0.0)},
        {ALG_PARAM_ID_INIT_ENTROPY_COEFF,       std::make_pair(1.0e-4,  1.0e4)},
        {ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT,    std::make_pair(1.0e0,  1.0e4)},
    },
};
