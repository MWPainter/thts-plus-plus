#pragma once

#include "main_aux/configs/hpopt_config_map.h"

#include "main_aux/configs/constants_algorithms.h"
#include "main_aux/configs/constants_experiments.h"
#include "main_aux/configs/constants_hpopt.h"

static const std::vector<HpoptConfigMap> HPOPT_CONFIG_320 =
{
    // xpr params
    {
        {XPR_OR_ALG_ID_TAG,                             HPOPT_PARAMS_ID_TAG},
        {XPR_PARAM_ID_NAME,                             "320_hpopt_frozen_lake_stoch_dense"},
        {XPR_PARAM_ID_ENV,                              ENV_ID_SLIPPY_FROZEN_LAKE_D_4x4},
        {XPR_PARAM_ID_MCTS_MODE,                        false},
        {XPR_PARAM_ID_GRAPH_SEARCH,                     true},
        {XPR_PARAM_ID_MAX_TRIAL_LENGTH,                 100},
        {XPR_PARAM_ID_RUNTIME_BOUNDED,                  false},
        {XPR_PARAM_ID_TERMINATION_BOUND,                10000},
        // {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,            2},
        {XPR_PARAM_ID_SEARCH_THREADS,                   16},
        {XPR_PARAM_ID_EVAL_DELTA,                       10000},
        {XPR_PARAM_ID_EVAL_ROLLOUTS,                    1024},
        {XPR_PARAM_ID_EVAL_THREADS,                     16},
        {HPOPT_PARAM_ID_MIN_REPEATS,                    10},
        {HPOPT_PARAM_ID_ESTIMATE_CONFIDENCE_THRESHOLD,  1.0},
        {HPOPT_PARAM_ID_BAYESOPT_TOTAL_SAMPLES,         200},
        {HPOPT_PARAM_ID_BAYESOPT_INIT_RAND_SAMPLES,     10},
        {HPOPT_PARAM_ID_BAYESOPT_RELEARN_FREQ,          20},
    },
    // uct params
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_UCT},
        {ALG_PARAM_ID_BIAS, std::make_pair(0.001,   1000.0)},
    },
    // max uct params
    {
        {XPR_OR_ALG_ID_TAG, ALG_ID_MAX_UCT},
        {ALG_PARAM_ID_BIAS, std::make_pair(0.001,   1000.0)},
    },
    // hmcts params
    {
        {XPR_OR_ALG_ID_TAG,         ALG_ID_HMCTS},
        {ALG_PARAM_ID_BIAS,         std::make_pair(1.0e-3,  1.0e4)},
        {ALG_PARAM_ID_UCT_BUDGET,   std::make_pair(1.0,     5000.0)},
    },
    // ments params
    {
        {XPR_OR_ALG_ID_TAG,             ALG_ID_MENTS},
        {ALG_PARAM_ID_INIT_TEMP,        std::make_pair(1.0e-4,  1.0e4)},
        {ALG_PARAM_ID_TEMP_DECAY_RATE,  std::make_pair(1.0e-4,  1.0e4)},
        {ALG_PARAM_ID_EPSILON,          std::make_pair(1.0e-6,  1.0e0)},
        {ALG_PARAM_ID_DEFAULT_Q_VALUE,  std::make_pair(0.0,  0.0)},
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
