#pragma once

#include "thts_env.h"
#include "thts_logger.h"
#include "thts_manager.h"

#include <memory>
#include <string>
#include <unordered_map>

// env ids
static const std::string DCHAIN_ENV_ID = "dchain_env";
static const std::string FL_ENV_ID = "frozen_lake_env";
static const std::string SAILING_ENV_ID = "sailing_env";

// chain env instance ids
static const std::string D_10_ID = "10-1.0";
static const std::string D_10_HALF_ID = "10-0.5";
static const std::string D_20_ID = "20-1.0";
static const std::string D_20_HALF_ID = "20-0.5";

// fl env instance ids
static const std::string FL_8x8 = "FL_8x8";
static const std::string FL_8x8_TEST = "FL_8x8_test";
static const std::string FL_8x16_TEST = "FL_8x16_test";


// sailing env instance ids
static const std::string S_5_ID = "5";
static const std::string S_7_ID = "7";
static const std::string S_10_ID = "10";

// expr ids
static const std::string DEBUG_EXPR_ID = "000";

static const std::string D001_LEN10 = "001_len_10";    
static const std::string D002_LEN20 = "002_len_20_ments_temp_search"; 
static const std::string D003_LEN20 = "003_len_20";     

static const std::string FL001_8_HPS = "001_FL8_HPS";
static const std::string FL002_8_TEST = "002_FL8_test";
static const std::string FL003_16_TEST = "003_FL16_test";

static const std::string S001_5 = "001_S5_TEST";
static const std::string S002_5 = "002_S5_HPS";
static const std::string S001_7 = "001_S7_TEST";
static const std::string S002_7 = "002_S7_HPS";
static const std::string S001_10 = "001_S10_TEST";
static const std::string S002_10 = "002_S10_HPS";

// alg ids
static const std::string ALG_ID_UCT = "uct";
static const std::string ALG_ID_PUCT = "puct";
static const std::string ALG_ID_MENTS = "ments";
static const std::string ALG_ID_DBMENTS = "db-ments";
static const std::string ALG_ID_DENTS = "dents";
static const std::string ALG_ID_DBDENTS = "db-dents";
static const std::string ALG_ID_RENTS = "rents";
static const std::string ALG_ID_TENTS = "tents";
static const std::string ALG_ID_EST = "est";

// param ids
static const std::string PARAMS_ID_UCT_BIAS = "bias";
static const std::string PARAMS_ID_MENTS_TEMP = "temp";
static const std::string PARAMS_ID_MENTS_EPSILON = "epsilon";
static const std::string PARAMS_ID_MENTS_DEFAULT_Q_VALUE = "default_q_value";
static const std::string PARAMS_ID_DENTS_INIT_TEMP = "value_temp_init";


namespace thts {
    /**
     * Struct to wrap all the params for a eval run
     * 
     * Member variables:
     *      env_id: A string id for an environment type
     *      env_instance_id: A string id for an instance of this env 
     *      expr_id: An id for the current experiment being run
     *      alg_id: An id to identify the algorithm to use for this run
     *      alg_params: An unordered map giving the params to pass to the algorithm
     *      num_trials: The number of trials to run for this experiment
     *      max_trial_length: The maximum trial length to use for the run
     *      trials_log_delta: The frequency of logging/running mc eval to use
     *      mc_eval_trials_delta: How often to run the mc eval
     *      rollouts_per_mc_eval: How many trials to use for mc evals   
     *      num_repeats: The number of times that this run should be repeated
     *      num_threads: The number of threads to use tree search
     *      eval_threads: The number of threads to use in mc evals
    */
    struct RunID {
        public:
            std::string env_id;
            std::string env_instance_id;
            std::string expr_id;
            std::string alg_id;
            std::unordered_map<std::string, double> alg_params;
            int num_trials;
            int max_trial_length;
            int trials_log_delta;
            int mc_eval_trials_delta;
            int rollouts_per_mc_eval;
            int num_repeats;
            int num_threads;
            int eval_threads;

            /**
             * Default constructor
            */
            RunID();

            /**
             * Initialised constructor
            */
            RunID(
                std::string env_id,
                std::string env_instance_id,
                std::string expr_id,
                std::string alg_id,
                std::unordered_map<std::string, double> alg_params,
                int num_trials,
                int max_trial_length,
                int trials_log_delta,
                int mc_eval_trials_delta,
                int rollouts_per_mc_eval,
                int num_repeats,
                int num_threads,
                int eval_threads);

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

            /**
             * Returns logger to use with search
            */
            std::shared_ptr<ThtsLogger> get_logger();
    };

    /**
     * Get a list of run id's from an experiment id
    */
    std::shared_ptr<std::vector<RunID>> get_run_ids_from_expr_id(std::string expr_id);
}