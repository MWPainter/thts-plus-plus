#pragma once

#include "thts_env.h"
#include "thts_manager.h"
#include "thts_decision_node.h"

#include <ctime>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "bayesopt/bayesopt.hpp"
#include "bayesopt/parameters.hpp"



// ---------------------------------------------------------------------------------------------------------------------
// Algorithm config
// ---------------------------------------------------------------------------------------------------------------------

// alg ids 
static const std::string UCT_ALG_ID = "uct";
static const std::string MAX_UCT_ALG_ID = "maxuct";
static const std::string MENTS_ALG_ID = "ments";
static const std::string BTS_ALG_ID = "bts";
static const std::string DENTS_ALG_ID = "dents";
static const std::string RENTS_ALG_ID = "rents";
static const std::string TENTS_ALG_ID = "tents";
static const std::string HMCTS_ALG_ID = "hmcts";

// param ids
static const std::string ADAPTIVE_BIAS_PARAM_ID = "adaptive_bias";      // adaptive bias (uct, etc)
static const std::string BIAS_PARAM_ID = "bias";                        // bias param (uct, etc)
static const std::string NORMALISE_Q_VALUES_PARAM_ID = "normalise_q_values"; // normalise q values (boltzmann search algorithms)
static const std::string TEMP_PARAM_ID = "temp";                        // temp (scale) param (ments, bts, dents, etc) (if using decay fn, then f(x) -> c*f(x))
static const std::string DECAY_FN_PARAM_ID = "decay_fn";                // temp decay fn (f(x))
static const std::string DECAY_FN_SCALE_PARAM_ID = "decay_fn_scale";    // f(x) -> f(c*x), f = decay fn
static const std::string ENTROPY_COEFF_PARAM_ID = "entropy_coeff";               // vertical scale for entropy decay fn (f(x) -> c*f(x))
static const std::string ENTROPY_DECAY_FN_PARAM_ID = "entropy_decay_fn";
static const std::string ENTROPY_DECAY_FN_SCALE_PARAM_ID = "decay_fn_scale"; // f(x) -> f(c*x), f = entropy decay fn
static const std::string EPSILON_PARAM_ID = "epsilon";                  // exploration param for stochastic policies
static const std::string DEFAULT_Q_VALUE_PARAM_ID = "default_q_value"; // default value of Q(s,a) for unseen state action pairs
static const std::string UCT_BUDGET_PARAM_ID = "uct_budget"; // hmcts's uct budget param

// param ids - decay fn options
enum DECAY_FN_VALUES {
    DECAY_FN_CONST = 0,
    DECAY_FN_INV_SQRT = 1,
    DECAY_FN_INV_LOG = 2,
};

// maps alg ids to the param ids that are relevent for it (there is overlap for example many algs use a temp param)
static const std::unordered_map<std::string,std::vector<std::string>> RELEVANT_PARAM_IDS =
{
    {UCT_ALG_ID,
        {
            ADAPTIVE_BIAS_PARAM_ID,
            BIAS_PARAM_ID,
        },
    },
    {MAX_UCT_ALG_ID,
        {
            ADAPTIVE_BIAS_PARAM_ID,
            BIAS_PARAM_ID,
        },
    },
    {MENTS_ALG_ID,
        {
            NORMALISE_Q_VALUES_PARAM_ID,
            TEMP_PARAM_ID,
            EPSILON_PARAM_ID,
            DEFAULT_Q_VALUE_PARAM_ID,
        },
    },
    {BTS_ALG_ID,
        {
            NORMALISE_Q_VALUES_PARAM_ID,
            TEMP_PARAM_ID,
            DECAY_FN_PARAM_ID,
            DECAY_FN_SCALE_PARAM_ID,
            EPSILON_PARAM_ID,
            DEFAULT_Q_VALUE_PARAM_ID,
        },
    },
    {DENTS_ALG_ID,
        {
            NORMALISE_Q_VALUES_PARAM_ID,
            TEMP_PARAM_ID,
            DECAY_FN_PARAM_ID,
            DECAY_FN_SCALE_PARAM_ID,
            ENTROPY_COEFF_PARAM_ID,
            ENTROPY_DECAY_FN_PARAM_ID,
            ENTROPY_DECAY_FN_SCALE_PARAM_ID,
            EPSILON_PARAM_ID,
            DEFAULT_Q_VALUE_PARAM_ID,
        },
    },
    {RENTS_ALG_ID,
        {
            NORMALISE_Q_VALUES_PARAM_ID,
            TEMP_PARAM_ID,
            EPSILON_PARAM_ID,
            DEFAULT_Q_VALUE_PARAM_ID,
        },
    },
    {TENTS_ALG_ID,
        {
            NORMALISE_Q_VALUES_PARAM_ID,
            TEMP_PARAM_ID,
            EPSILON_PARAM_ID,
            DEFAULT_Q_VALUE_PARAM_ID,
        },
    },
    {HMCTS_ALG_ID,
        {
            ADAPTIVE_BIAS_PARAM_ID,
            BIAS_PARAM_ID,
            UCT_BUDGET_PARAM_ID,
        },
    },
};

// List of boolean param ids (for hyperparam opt)
static const std::unordered_set<std::string> BOOLEAN_PARAM_IDS =
{
    ADAPTIVE_BIAS_PARAM_ID,
    NORMALISE_Q_VALUES_PARAM_ID,
};

// List of int param ids (for hyperparam opt)
static const std::unordered_set<std::string> INTEGER_PARAM_IDS =
{
    DECAY_FN_PARAM_ID,
    ENTROPY_DECAY_FN_PARAM_ID,
    UCT_BUDGET_PARAM_ID,
};

// List of params to use a log scale in BayesOpt
static const std::unordered_set<std::string> LOG_SCALE_PARAM_IDS =
{
    BIAS_PARAM_ID,
    TEMP_PARAM_ID,
    DECAY_FN_SCALE_PARAM_ID,
    ENTROPY_COEFF_PARAM_ID,
    ENTROPY_DECAY_FN_SCALE_PARAM_ID,
};


// ---------------------------------------------------------------------------------------------------------------------
// Environment config
// ---------------------------------------------------------------------------------------------------------------------

// env ids
static const std::string D_CHAIN_10_ENV_ID = "dchain(D=10,R=1.0)";
static const std::string MOD_D_CHAIN_10_ENV_ID = "dchain(D=10,R=0.5)";
static const std::string ENTROPY_TRAP_10_ENV_ID = "dchain(D=10,H=10)";
static const std::string ENTROPY_TRAP_15_ENV_ID = "dchain(D=15,H=15)";
static const std::string FROZEN_LAKE_NO_HOLE_DENSE_ENV_ID = "frozen_lake_no_hole_dense";
static const std::string FROZEN_LAKE_NO_HOLE_SPARSE_LEN_ENV_ID = "frozen_lake_no_hole_sparse_len";
static const std::string FROZEN_LAKE_NO_HOLE_SPARSE_DISCOUNTED_ENV_ID = "frozen_lake_no_hole_sparse_discounted";
static const std::string FROZEN_LAKE_D_8x8_ENV_ID = "frozen_lake_(map=8x8,dense)";
static const std::string FROZEN_LAKE_S_8x8_ENV_ID = "frozen_lake_(map=8x8,sparse)";
static const std::string FROZEN_LAKE_D_8x16_ENV_ID = "frozen_lake_(map=8x16,dense)";
static const std::string FROZEN_LAKE_S_8x16_ENV_ID = "frozen_lake_(map=8x16,sparse)";
static const std::string FROZEN_LAKE_D_16x16_ENV_ID = "frozen_lake_(map=16x16,dense)";
static const std::string FROZEN_LAKE_S_16x16_ENV_ID = "frozen_lake_(map=16x16,sparse)";
static const std::string SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID = "slippy_frozen_lake_(map=4x4,dense)";
static const std::string SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID = "slippy_frozen_lake_(map=4x4,sparse)";
static const std::string SLIPPY_FROZEN_LAKE_D_5x5_ENV_ID = "slippy_frozen_lake_(map=5x5,dense)";
static const std::string SLIPPY_FROZEN_LAKE_S_5x5_ENV_ID = "slippy_frozen_lake_(map=5x5,sparse)";
static const std::string SLIPPY_FROZEN_LAKE_D_6x6_ENV_ID = "slippy_frozen_lake_(map=6x6,dense)";
static const std::string SLIPPY_FROZEN_LAKE_S_6x6_ENV_ID = "slippy_frozen_lake_(map=6x6,sparse)";
static const std::string SAILING_ENV_NORTH_ID = "sailing_north";
static const std::string SAILING_ENV_SOUTH_EAST_ID = "sailing_south_east";

// env ids - python envs (non gym envs that need the python )interpreter
// TODO: any python envs

static const std::unordered_set<std::string> PY_ENVS =
{
};

// env ids - gym envs (envs that are python gym envs)
static const std::string TAXI_GYM_ENV_ID = "Taxi-v3"; // https://gymnasium.farama.org/environments/toy_text/taxi/

static const std::unordered_set<std::string> GYM_ENVS =
{
    TAXI_GYM_ENV_ID,
};

// env ids - max trial length
static const std::unordered_map<std::string,int> ENV_ID_MAX_TRIAL_LEN = 
{
    {D_CHAIN_10_ENV_ID,         10000},
    {MOD_D_CHAIN_10_ENV_ID,     10000},
    {ENTROPY_TRAP_10_ENV_ID,    10000},
    {ENTROPY_TRAP_15_ENV_ID,    10000},
    {FROZEN_LAKE_NO_HOLE_DENSE_ENV_ID,50},
    {FROZEN_LAKE_NO_HOLE_SPARSE_LEN_ENV_ID,50},
    {FROZEN_LAKE_NO_HOLE_SPARSE_DISCOUNTED_ENV_ID,50},
    {FROZEN_LAKE_D_8x8_ENV_ID,    10000},
    {FROZEN_LAKE_S_8x8_ENV_ID,    10000},
    {FROZEN_LAKE_D_8x16_ENV_ID,    10000},
    {FROZEN_LAKE_S_8x16_ENV_ID,    10000},
    {FROZEN_LAKE_D_16x16_ENV_ID,    10000},
    {FROZEN_LAKE_S_16x16_ENV_ID,    10000},
    {SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID,    10000},
    {SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID,    10000},
    {SLIPPY_FROZEN_LAKE_D_5x5_ENV_ID,    10000},
    {SLIPPY_FROZEN_LAKE_S_5x5_ENV_ID,    10000},
    {SLIPPY_FROZEN_LAKE_D_6x6_ENV_ID,    10000},
    {SLIPPY_FROZEN_LAKE_S_6x6_ENV_ID,    10000},
    {SAILING_ENV_NORTH_ID,      10000},
    {SAILING_ENV_SOUTH_EAST_ID, 10000},
};

static const std::unordered_set<std::string> DET_ENVS = 
{
    D_CHAIN_10_ENV_ID,
    MOD_D_CHAIN_10_ENV_ID,
    ENTROPY_TRAP_10_ENV_ID,
    FROZEN_LAKE_NO_HOLE_DENSE_ENV_ID,
    FROZEN_LAKE_NO_HOLE_SPARSE_LEN_ENV_ID,
    FROZEN_LAKE_NO_HOLE_SPARSE_DISCOUNTED_ENV_ID,
    FROZEN_LAKE_D_8x8_ENV_ID,
    FROZEN_LAKE_S_8x8_ENV_ID,
    FROZEN_LAKE_D_8x16_ENV_ID,
    FROZEN_LAKE_S_8x16_ENV_ID,
    FROZEN_LAKE_D_16x16_ENV_ID,
    FROZEN_LAKE_S_16x16_ENV_ID,
};



// ---------------------------------------------------------------------------------------------------------------------
// Experiment config
// ---------------------------------------------------------------------------------------------------------------------

// expr ids - debug
static const std::string DEBUG_EXPR_ID = "000_debug";

// expr ids - supp experiments (1xx + 2xx + 3xx)
// supp experiments = showing how performance varies with parameters etc
static const std::string SUPP_100_DCHAIN_10_TEMP_EXPR_ID = "100_supp_dchain_temp_vary";
static const std::string SUPP_101_MOD_DCHAIN_10_TEMP_EXPR_ID = "101_supp_mod_dchain_temp_vary";
static const std::string SUPP_102_ENTROPY_TRAP_10_TEMP_EXPR_ID = "102_supp_entropy_temp_10_vary";
static const std::string SUPP_103_ENTROPY_TRAP_15_TEMP_EXPR_ID = "103_supp_entropy_temp_15_vary";
// TODO: what about the exploration param - make an expr or two for this.
static const std::string SUPP_110_UCT_ON_FL_DENSE = "110_uct_on_fl_dense";
static const std::string SUPP_111_UCT_ON_FL_SPARSE_LEN = "111_uct_on_fl_sparse_len";
static const std::string SUPP_112_UCT_ON_FL_SPARSE_DISCOUNTED = "112_uct_on_fl_sparse_discounted";

// expr ids - toy experiments (4xx + 5xx)
// toy experiments = running experiments on the toy envs
static const std::string TOY_XXX_EXPR_ID = "400_xxx";

// expr ids - rerun experiments (6xx + 7xx = hyperparam, 8xx + 9xx = eval)
// rerunning experiments = repeating the experiments with hyperparam tuning now
static const std::string HP_OPT_600_UCT_EXPR_ID =       "600_hp_opt_uct";
static const std::string HP_OPT_610_MAX_UCT_EXPR_ID =   "610_hp_opt_max_uct";
static const std::string HP_OPT_620_MENTS_EXPR_ID =     "620_hp_opt_ments";
static const std::string HP_OPT_630_BTS_EXPR_ID =       "630_hp_opt_bts";
static const std::string HP_OPT_640_DENTS_EXPR_ID =     "640_hp_opt_dents";
static const std::string HP_OPT_650_RENTS_EXPR_ID =     "650_hp_opt_rents";
static const std::string HP_OPT_660_TENTS_EXPR_ID =     "660_hp_opt_tents";
static const std::string HP_OPT_670_HMCTS_EXPR_ID =     "670_hp_opt_hmcts";

static const std::string HP_OPT_601_UCT_EXPR_ID =       "601_hp_opt_uct";
static const std::string HP_OPT_611_MAX_UCT_EXPR_ID =   "611_hp_opt_max_uct";
static const std::string HP_OPT_621_MENTS_EXPR_ID =     "621_hp_opt_ments";
static const std::string HP_OPT_631_BTS_EXPR_ID =       "631_hp_opt_bts";
static const std::string HP_OPT_641_DENTS_EXPR_ID =     "641_hp_opt_dents";
static const std::string HP_OPT_651_RENTS_EXPR_ID =     "651_hp_opt_rents";
static const std::string HP_OPT_661_TENTS_EXPR_ID =     "661_hp_opt_tents";
static const std::string HP_OPT_671_HMCTS_EXPR_ID =     "671_hp_opt_hmcts";

static const std::string HP_OPT_602_UCT_EXPR_ID =       "602_hp_opt_uct";
static const std::string HP_OPT_612_MAX_UCT_EXPR_ID =   "612_hp_opt_max_uct";
static const std::string HP_OPT_622_MENTS_EXPR_ID =     "622_hp_opt_ments";
static const std::string HP_OPT_632_BTS_EXPR_ID =       "632_hp_opt_bts";
static const std::string HP_OPT_642_DENTS_EXPR_ID =     "642_hp_opt_dents";
static const std::string HP_OPT_652_RENTS_EXPR_ID =     "652_hp_opt_rents";
static const std::string HP_OPT_662_TENTS_EXPR_ID =     "662_hp_opt_tents";
static const std::string HP_OPT_672_HMCTS_EXPR_ID =     "672_hp_opt_hmcts";

static const std::string HP_OPT_603_UCT_EXPR_ID =       "603_hp_opt_uct";
static const std::string HP_OPT_613_MAX_UCT_EXPR_ID =   "613_hp_opt_max_uct";
static const std::string HP_OPT_623_MENTS_EXPR_ID =     "623_hp_opt_ments";
static const std::string HP_OPT_633_BTS_EXPR_ID =       "633_hp_opt_bts";
static const std::string HP_OPT_643_DENTS_EXPR_ID =     "643_hp_opt_dents";
static const std::string HP_OPT_653_RENTS_EXPR_ID =     "653_hp_opt_rents";
static const std::string HP_OPT_663_TENTS_EXPR_ID =     "663_hp_opt_tents";
static const std::string HP_OPT_673_HMCTS_EXPR_ID =     "673_hp_opt_hmcts";

static const std::string EVAL_XXX_EXPR_ID = "800_eval_xxx_env_xxx";

// env id lookup - helper dict to lookup env ids from hp opt experiment ids
static const std::unordered_map<std::string,std::string> HP_OPT_EXPR_ID_TO_ENV_ID =
{
    {HP_OPT_600_UCT_EXPR_ID,                    FROZEN_LAKE_D_8x8_ENV_ID},
    {HP_OPT_610_MAX_UCT_EXPR_ID,                FROZEN_LAKE_D_8x8_ENV_ID},
    {HP_OPT_620_MENTS_EXPR_ID,                  FROZEN_LAKE_D_8x8_ENV_ID},
    {HP_OPT_630_BTS_EXPR_ID,                    FROZEN_LAKE_D_8x8_ENV_ID},
    {HP_OPT_640_DENTS_EXPR_ID,                  FROZEN_LAKE_D_8x8_ENV_ID},
    {HP_OPT_650_RENTS_EXPR_ID,                  FROZEN_LAKE_D_8x8_ENV_ID},
    {HP_OPT_660_TENTS_EXPR_ID,                  FROZEN_LAKE_D_8x8_ENV_ID},
    {HP_OPT_670_HMCTS_EXPR_ID,                  FROZEN_LAKE_D_8x8_ENV_ID},

    {HP_OPT_601_UCT_EXPR_ID,                    FROZEN_LAKE_S_8x8_ENV_ID},
    {HP_OPT_611_MAX_UCT_EXPR_ID,                FROZEN_LAKE_S_8x8_ENV_ID},
    {HP_OPT_621_MENTS_EXPR_ID,                  FROZEN_LAKE_S_8x8_ENV_ID},
    {HP_OPT_631_BTS_EXPR_ID,                    FROZEN_LAKE_S_8x8_ENV_ID},
    {HP_OPT_641_DENTS_EXPR_ID,                  FROZEN_LAKE_S_8x8_ENV_ID},
    {HP_OPT_651_RENTS_EXPR_ID,                  FROZEN_LAKE_S_8x8_ENV_ID},
    {HP_OPT_661_TENTS_EXPR_ID,                  FROZEN_LAKE_S_8x8_ENV_ID},
    {HP_OPT_671_HMCTS_EXPR_ID,                  FROZEN_LAKE_S_8x8_ENV_ID},

    {HP_OPT_602_UCT_EXPR_ID,                    SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID},
    {HP_OPT_612_MAX_UCT_EXPR_ID,                SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID},
    {HP_OPT_622_MENTS_EXPR_ID,                  SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID},
    {HP_OPT_632_BTS_EXPR_ID,                    SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID},
    {HP_OPT_642_DENTS_EXPR_ID,                  SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID},
    {HP_OPT_652_RENTS_EXPR_ID,                  SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID},
    {HP_OPT_662_TENTS_EXPR_ID,                  SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID},
    {HP_OPT_672_HMCTS_EXPR_ID,                  SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID},

    {HP_OPT_603_UCT_EXPR_ID,                    SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID},
    {HP_OPT_613_MAX_UCT_EXPR_ID,                SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID},
    {HP_OPT_623_MENTS_EXPR_ID,                  SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID},
    {HP_OPT_633_BTS_EXPR_ID,                    SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID},
    {HP_OPT_643_DENTS_EXPR_ID,                  SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID},
    {HP_OPT_653_RENTS_EXPR_ID,                  SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID},
    {HP_OPT_663_TENTS_EXPR_ID,                  SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID},
    {HP_OPT_673_HMCTS_EXPR_ID,                  SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID},
};

// list of all expr ids (for helper to lookup expr id from a prefix (just the number))
static const std::unordered_set<std::string> ALL_EXPR_IDS = 
{
    DEBUG_EXPR_ID,
    SUPP_100_DCHAIN_10_TEMP_EXPR_ID,
    SUPP_101_MOD_DCHAIN_10_TEMP_EXPR_ID,
    SUPP_102_ENTROPY_TRAP_10_TEMP_EXPR_ID,
    SUPP_103_ENTROPY_TRAP_15_TEMP_EXPR_ID,
    SUPP_110_UCT_ON_FL_DENSE,
    SUPP_111_UCT_ON_FL_SPARSE_LEN,
    SUPP_112_UCT_ON_FL_SPARSE_DISCOUNTED,

    HP_OPT_600_UCT_EXPR_ID,      
    HP_OPT_610_MAX_UCT_EXPR_ID,    
    HP_OPT_620_MENTS_EXPR_ID,      
    HP_OPT_630_BTS_EXPR_ID,                    
    HP_OPT_640_DENTS_EXPR_ID,                  
    HP_OPT_650_RENTS_EXPR_ID,                  
    HP_OPT_660_TENTS_EXPR_ID,                  
    HP_OPT_670_HMCTS_EXPR_ID,

    HP_OPT_601_UCT_EXPR_ID,      
    HP_OPT_611_MAX_UCT_EXPR_ID,    
    HP_OPT_621_MENTS_EXPR_ID,      
    HP_OPT_631_BTS_EXPR_ID,                    
    HP_OPT_641_DENTS_EXPR_ID,                  
    HP_OPT_651_RENTS_EXPR_ID,                  
    HP_OPT_661_TENTS_EXPR_ID,                  
    HP_OPT_671_HMCTS_EXPR_ID,

    HP_OPT_602_UCT_EXPR_ID,      
    HP_OPT_612_MAX_UCT_EXPR_ID,    
    HP_OPT_622_MENTS_EXPR_ID,      
    HP_OPT_632_BTS_EXPR_ID,                    
    HP_OPT_642_DENTS_EXPR_ID,                  
    HP_OPT_652_RENTS_EXPR_ID,                  
    HP_OPT_662_TENTS_EXPR_ID,                  
    HP_OPT_672_HMCTS_EXPR_ID,

    HP_OPT_603_UCT_EXPR_ID,      
    HP_OPT_613_MAX_UCT_EXPR_ID,    
    HP_OPT_623_MENTS_EXPR_ID,      
    HP_OPT_633_BTS_EXPR_ID,                    
    HP_OPT_643_DENTS_EXPR_ID,                  
    HP_OPT_653_RENTS_EXPR_ID,                  
    HP_OPT_663_TENTS_EXPR_ID,                  
    HP_OPT_673_HMCTS_EXPR_ID,                  
};



// ---------------------------------------------------------------------------------------------------------------------
// RunID and Hyperparam Opt class definitions
// ---------------------------------------------------------------------------------------------------------------------

namespace thts {
    /**
     * Struct to wrap all the params for a eval run
     * 
     * Member variables - env/alg params:
     *      env_id: A string id for an environment instance
     *      expr_id: An id for the current experiment being run
     *      expr_timestamp: A timestamp to mark expr_id with (so can rerun same expr without overwriting results)
     *      alg_params: dictionary of alg params below
     * 
     *      <values for all of the params that algorithms use>
     * 
     * Member variables - tree search params:
     *      eval_wrt_time: If we want to eval with respect to runtime (or max number of trials)
     *      search_runtime: The total runtime to use for each search (in seconds, or #trials)
     *      max_trial_length: The maximum trial length to use for the run
     *      eval_delta: The frequency of logging/running mc eval to use (in seconds, or #trials)
     *      rollouts_per_mc_eval: How many trials to use for mc evals   
     *      num_repeats: The number of times that this run should be repeated
     *      num_threads: The number of threads to use tree search
     *      eval_threads: The number of threads to use in mc evals
     *      num_envs: The number of environments for ThtsManager to duplicate
    */
    struct RunID {
        public:
            std::string env_id;
            std::string expr_id;
            std::time_t expr_timestamp;
            std::string alg_id;

            std::unordered_map<std::string, double> alg_params;

            bool adaptive_bias;
            double bias;
            int hmcts_uct_budget;

            bool normalise_q_values;
            double temp;
            int decay_fn;
            double decay_fn_scale;

            double entropy_coeff;
            int entropy_decay_fn;
            double entropy_decay_fn_scale;
            
            bool eval_wrt_time;
            double search_runtime;
            double eval_delta;
            int rollouts_per_mc_eval;
            int max_trial_length;
            int num_repeats;
            int num_threads;
            int eval_threads;
            int num_envs;

            /**
             * Default constructor
            */
            RunID();

            /**
             * Initialised constructor
            */
            RunID(
                std::string env_id,
                std::string expr_id,
                std::time_t expr_timestamp,
                std::string alg_id,
                std::unordered_map<std::string, double>& alg_params,
                bool eval_wrt_time,
                double search_runtime,
                double eval_delta,
                int rollouts_per_mc_eval,
                int max_trial_length,
                int num_repeats,
                int num_threads,
                int eval_threads);

            /**
             * A unique results directory for each RunID
             */
            std::string get_results_dir();

            /**
             * Returns if the env we are using is a python env
            */
            bool is_python_env();

            /**
             * Returns an instance of ThtsEnv to use for this run
            */
            std::shared_ptr<ThtsEnv> get_env();

            /**
             * Returns and instance of ThtsManager to use for this run
            */
            std::shared_ptr<ThtsManager> get_thts_manager(std::shared_ptr<ThtsEnv> env);

            /**
             * Returns a root node to use for search given these params
            */
            std::shared_ptr<ThtsDNode> get_root_search_node(
                std::shared_ptr<ThtsEnv> env, std::shared_ptr<ThtsManager> manager);
    };

    /**
     * Get a list of run id's from an experiment id
    */
    std::shared_ptr<std::vector<RunID>> get_run_ids_from_expr_id_prefix(std::string expr_id_prefix);

    /**
     * Class for running hyperparam optimisation
     * 
     * 'alg_param_ids' 
     *      is used to map between vectors (used in bayesopt) and param ids
     * 'alg_param_min_max[param_id]' 
     *      specifies the maximum and minimum values to use in bayesopt for param id
     *      N.B. min and max can be arbitrary for a boolean value, but may as well be 0.0, and 1.0
     *          and for integer value, we will sample in the *integer* range [min,max)
     */
    class HyperparamOptimiser : public bayesopt::ContinuousModel
    {
        public:
            int num_hyperparams;

            std::string env_id;
            std::string expr_id;
            std::time_t expr_timestamp;
            std::string alg_id;

            std::vector<std::string> alg_param_ids;
            std::unordered_map<std::string, std::pair<double,double>> alg_params_min_max;

            bool eval_wrt_time;
            double search_runtime;
            int max_trial_length;
            double eval_delta;
            int rollouts_per_mc_eval;
            int num_repeats;
            int num_threads;
            int eval_threads;
            int num_envs;

            double best_eval;
            std::unordered_map<std::string, double> best_alg_params;

            std::ofstream &results_fs;
            int hp_opt_iter;
            
            HyperparamOptimiser(
                std::string env_id,
                std::string expr_id,
                std::time_t expr_timestamp,
                std::string alg_id,
                std::unordered_map<std::string, std::pair<double,double>> alg_params_min_max,
                bool eval_wrt_time,
                double search_runtime,
                int max_trial_length,
                double eval_delta,
                int rollouts_per_mc_eval,
                int num_repeats,
                int num_threads,
                int eval_threads,
                bayesopt::Parameters params,
                std::ofstream &results_fs);

            bool is_python_env();

            virtual std::unordered_map<std::string, double> get_alg_params_from_bayesopt_vec(bayesopt::vectord vec);

            bool get_bool_val_from_cts_sample(double sample_val, int min, int max);

            int get_int_val_from_cts_sample(double sample_val, int min, int max);

            virtual double evaluateSample(const bayesopt::vectord &query) override;

            void write_header();

        private:
            void write_eval_line(std::unordered_map<std::string,double> alg_params, double eval);

        public:
            void write_best_eval();
    };

    /**
     * Creates and returns a hyperparamters optimiser from experiment id
    */
    std::shared_ptr<HyperparamOptimiser> get_hyperparam_optimiser_from_expr_id(
        std::string expr_id, std::time_t expr_timestamp, std::ofstream &hp_opt_fs);

    /**
     * Lookup expr_id from prefix
     */
    std::string lookup_expr_id_from_prefix(std::string expr_id_prefix);

    /**
     * A unique results directory for each RunID
     */
    std::string get_results_dir(RunID& run_id);

    /**
     * Checks if env corresponding to 'env_id' is a python env
     */
    bool is_python_env(std::string env_id);

    /**
     * Create the env corresponding to 'env_id' and return is
     */
    std::shared_ptr<ThtsEnv> get_env(RunID& run_id);
}