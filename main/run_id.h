#pragma once

#include "mo/mo_thts_env.h"
#include "mo/mo_thts_manager.h"
#include "mo/mo_thts_decision_node.h"

#include <ctime>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/Dense>

#include "bayesopt/bayesopt.hpp"
#include "bayesopt/parameters.hpp"



// ---------------------------------------------------------------------------------------------------------------------
// Algorithm config
// ---------------------------------------------------------------------------------------------------------------------

// alg ids 
static const std::string UCT_ALG_ID = "uct";
static const std::string BTS_ALG_ID = "bts";
// TODO

// param ids
static const std::string BIAS_PARAM_ID = "bias";                        // bias param (uct, etc)
static const std::string TEMP_PARAM_ID = "temp";                        // temp param (ments, bts, dents, etc)
static const std::string DECAY_FN_PARAM_ID = "decay_fn";                // temp decay fn
static const std::string DECAY_FN_COEFF_PARAM_ID = "decay_fn_coeff";    // f(x) -> c*f(x), f = decay fn
static const std::string DECAY_FN_SCALE_PARAM_ID = "decay_fn_scale";    // f(x) -> f(c*x), f = decay fn
// TODO

// param ids - decay fn options
enum DECAY_FN_VALUES {
    DECAY_FN_CONST = 0,
    DECAY_FN_INV_SQRT = 1,
    DECAY_FN_INV_LOG = 2,
}

// maps alg ids to the param ids that are relevent for it (there is overlap for example many algs use a temp param)
static const std::unordered_map<std::string,std::vector<std::string>> RELEVANT_PARAM_IDS =
{
    {UCT_ALG_ID,
        {
            UCT_BIAS_PARAM_ID,
        },
    },
    {BTS_ALG_ID,
        {
            TEMP_PARAM_ID,
            DECAY_FN_PARAM_ID,
            DECAY_FN_COEFF_PARAM_ID,
            DECAY_FN_SCALE_PARAM_ID,
        },
    },
    // TODO
};

// List of boolean param ids (for hyperparam opt)
static const std::unordered_set<std::string> BOOLEAN_PARAM_IDS =
{
    // TODO
};

// List of int param ids (for hyperparam opt)
static const std::unordered_set<std::string> INTEGER_PARAM_IDS =
{
    DECAY_FN_PARAM_ID,
};



// ---------------------------------------------------------------------------------------------------------------------
// Environment config
// ---------------------------------------------------------------------------------------------------------------------

// env ids
static const std::string FROZEN_LAKE_4x4_ENV_ID = "frozen_lake_(map=4x4)";
static const std::string FROZEN_LAKE_8x8_ENV_ID = "frozen_lake_(map=8x8)";
// TODO

// env ids - python envs
// a list of envs that need the python interpreter
static const std::string XXX_PY_ENV_ID = "xxx";

static const std::unordered_set<std::string> PY_ENVS =
{
    // TODO
};

// env ids - gym envs
// a list of envs that are python gym envs
static const std::string TAXI_GYM_ENV_ID = "Taxi-v3"; // https://gymnasium.farama.org/environments/toy_text/taxi/

static const std::unordered_set<std::string> GYM_ENVS =
{
    TAXI_ENV_ID,
};

// env ids - max trial length
static const std::unordered_map<std::string,int> ENV_ID_MAX_TRIAL_LEN = 
{
    {FROZEN_LAKE_4x4_ENV_ID,    25},
    {FROZEN_LAKE_8x8_ENV_ID,    50},
    // TODO
};



// ---------------------------------------------------------------------------------------------------------------------
// Experiment config
// ---------------------------------------------------------------------------------------------------------------------

// expr ids - debug
static const std::string DEBUG_EXPR_ID = "000_debug";

// expr ids - supp experiments (1xx + 2xx + 3xx)
// supp experiments = showing how performance varies with parameters etc
static const std::string SUPP_XXX_EXPR_ID = "100_xxx";

// expr ids - toy experiments (4xx + 5xx)
// toy experiments = running experiments on the toy envs
static const std::string TOY_XXX_EXPR_ID = "400_xxx";

// expr ids - rerun experiments (6xx + 7xx = hyperparam, 8xx + 9xx = eval)
// rerunning experiments = repeating the experiments (minus go) plus a couple extra, wiv hyperparam tuning now
static const std::string HP_OPT_XXX_UCT_EXPR_ID = "600_hp_opt_xxx_env_xxx_alg";
static const std::string HP_OPT_XXX_BTS_EXPR_ID = "600_hp_opt_xxx_env_bts";
static const std::string EVAL_XXX_EXPR_ID = "800_eval_xxx_env_xxx";

// env id lookup - helper dict to lookup env ids from hp opt experiment ids
static const std::unordered_map<std::string,std::string> HP_OPT_EXPR_ID_TO_ENV_ID =
{
    {HP_OPT_XXX_UCT_EXPR_ID,                FROZEN_LAKE_4x4_ENV_ID},
    {HP_OPT_XXX_BTS_EXPR_ID,                FROZEN_LAKE_4x4_ENV_ID},
    // TODO
};

// list of all expr ids (for helper to lookup expr id from a prefix (just the number))
static const std::unordered_set<std::string> ALL_EXPR_IDS = 
{
    DEBUG_EXPR_ID,
    // TODO
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

            double bias;
            // TODO: add params for algorithms
            
            bool eval_wrt_time;
            double search_runtime;
            int max_trial_length;
            double eval_num_trials_delta;
            double eval_delta;
            int rollouts_per_mc_eval;
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
                int max_trial_length,
                double eval_delta,
                int rollouts_per_mc_eval,
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
            std::shared_ptr<ThtsManager> get_thts_manager(std::shared_ptr<MoThtsEnv> env);

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
    std::shared_ptr<MoThtsEnv> get_env(RunID& run_id);
}