#pragma once

#include "mo/mo_thts_env.h"
#include "mo/mo_thts_manager.h"
#include "mo/mo_thts_decision_node.h"

#include <ctime>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/Dense>

// env ids - debug
static const std::string DEBUG_ENV_1_ID = "debug_env_1"; // not stoch + 2 rew
static const std::string DEBUG_ENV_2_ID = "debug_env_2"; // stoch + 2 rew
static const std::string DEBUG_ENV_3_ID = "debug_env_3"; // not stoch + 4 rew
static const std::string DEBUG_ENV_4_ID = "debug_env_4"; // stoch + 4 rew
static const std::string DEBUG_PY_ENV_1_ID = "py_debug_env_1"; // not stoch + 2 rew
static const std::string DEBUG_PY_ENV_2_ID = "py_debug_env_2"; // stoch + 2 rew
static const std::string DEBUG_PY_ENV_3_ID = "py_debug_env_3"; // not stoch + 4 rew
static const std::string DEBUG_PY_ENV_4_ID = "py_debug_env_4"; // stoch + 4 rew

// env ids - toy/tree
// TODO: implement + integrate tree envs to demonstrate things

// env ids - mo gymnasium (discr obs / discr act)
static const std::string DST_ENV_ID = "deep-sea-treasure-v0";
static const std::string DST_CONC_ENV_ID = "deep-sea-treasure-concave-v0";
static const std::string DST_MIRR_ENV_ID = "deep-sea-treasure-mirrored-v0";
static const std::string RESOURCE_GATHER_ENV_ID = "resource-gathering-v0";
static const std::string BREAKABLE_BOTTLES_ENV_ID = "breakable-bottles-v0";
static const std::string FRUIT_TREE_ENV_ID = "fruit-tree-v0";
static const std::string FOUR_ROOM_ENV_ID = "four-room-v0";

// env ids - mo gymnasium (cts obs / discr act)
static const std::string MOUNTAIN_CAR_ENV_ID = "mo-mountaincar-v0";
static const std::string LUNAR_LANDER_ENV_ID = "mo-lunar-lander-v2";
static const std::string MINECART_ENV_ID = "minecart-v0";
static const std::string HIGHWAY_ENV_ID = "mo-highway-v0";
static const std::string HIGHWAY_FAST_ENV_ID = "mo-highway-fast-v0";

static const std::unordered_set<std::string> MO_GYM_ENVS =
{
     DST_ENV_ID,
     DST_CONC_ENV_ID,
     DST_MIRR_ENV_ID,
     RESOURCE_GATHER_ENV_ID,
     BREAKABLE_BOTTLES_ENV_ID,
     FRUIT_TREE_ENV_ID,
     FOUR_ROOM_ENV_ID,
     MOUNTAIN_CAR_ENV_ID,
     LUNAR_LANDER_ENV_ID,
     MINECART_ENV_ID,
     HIGHWAY_ENV_ID,
     HIGHWAY_FAST_ENV_ID,
};

// alg ids 
static const std::string CZT_ALG_ID = "czt";
static const std::string CHMCTS_ALG_ID = "chmcts";
static const std::string SMBTS_ALG_ID = "smbts";
static const std::string SMDENTS_ALG_ID = "smdents";

// expr ids - testing
static const std::string DEBUG_EXPR_ID = "000_debug";
static const std::string DEBUG_ENV_EXPR_ID = "001_debug_env_1";
static const std::string DEBUG_PY_ENV_EXPR_ID = "002_debug_py_env_1";
static const std::string DEBUG_ENV_EXPR_ID = "003_debug_env_2";
static const std::string DEBUG_PY_ENV_EXPR_ID = "004_debug_py_env_2";
static const std::string DEBUG_ENV_EXPR_ID = "005_debug_env_3";
static const std::string DEBUG_PY_ENV_EXPR_ID = "006_debug_py_env_3";
static const std::string DEBUG_ENV_EXPR_ID = "007_debug_env_4";
static const std::string DEBUG_PY_ENV_EXPR_ID = "008_debug_py_env_4";
static const std::string POC_DST_EXPR_ID = "009_poc_dst";
static const std::string POC_FT_EXPR_ID = "010_poc_ft";

// expr ids - toy/tree
// TODO: 1xx = hyperparam tuning 
// TODO: 2xx = eval

// expr ids - mo gymnasium
// TODO: 3xx = hyperparam tuning
// TODO: 4xx = eval

// param ids
static const std::string CZT_BIAS_PARAM_ID = "czt_bias";
static const std::string CZT_BALL_SPLIT_VISIT_THRESH_PARAM_ID = "czt_ball_split_visit_thresh";

static const std::string SM_L_INF_THRESH_PARAM_ID = "sm_l_inf_thresh";
static const std::string SM_MAX_DEPTH = "sm_max_depth";
static const std::string SM_SPLIT_VISIT_THRESH_PARAM_ID = "sm_split_visit_thresh";

static const std::string SMBTS_SEARCH_TEMP_PARAM_ID = "smbts_search_temp";
static const std::string SMBTS_EPSILON_PARAM_ID = "smbts_epsilon";
static const std::string SMBTS_SEARCH_TEMP_USE_DECAY_PARAM_ID = "smbts_use_search_temp_decay";
static const std::string SMBTS_SEARCH_TEMP_DECAY_VISITS_SCALE_PARAM_ID = "smbts_search_temp_decay_visits_scale";

static const std::string SMDENTS_ENTROPY_TEMP_INIT_PARAM_ID = "smdents_entropy_temp_init";
static const std::string SMDENTS_ENTROPY_TEMP_VISITS_SCALE_PARAM_ID = "smdents_entropy_temp_visits_scale";


namespace thts {
    /**
     * Struct to wrap all the params for a eval run
     * 
     * Member variables - env/alg params:
     *      env_id: A string id for an environment instance
     *      expr_id: An id for the current experiment being run
     *      expr_timestamp: A timestamp to mark expr_id with (so can rerun same expr without overwriting results)
     *      alg_params: dictionary of alg params below
     *      czt_bias: czt / chmcts param
     *      czt_ball_split_visit_thresh: czt / chmcts param
     *      sm_l_inf_thresh: simplex map param
     *      sm_max_depth: simplex map param
     *      sm_split_visit_thresh: simplex map param
     *      smbts_search_temp: smbts param
     *      smbts_epsilon: smbts param
     *      smbts_use_search_temp_decay: smbts param
     *      smbts_search_temp_decay_visits_scale: smbts param
     *      smdents_entropy_temp_init: smdents param
     *      smdents_entropy_temp_visits_scale: smdents param
     * 
     * Member variables - tree search params:
     *      search_runtime: The total runtime to use for each search (in seconds)
     *      max_trial_length: The maximum trial length to use for the run
     *      eval_delta: The frequency of logging/running mc eval to use (in seconds)
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

            double czt_bias;
            int czt_ball_split_visit_thresh;
            double sm_l_inf_thresh;
            int sm_max_depth;
            int sm_split_visit_thresh;
            double smbts_search_temp;
            double smbts_epsilon;
            bool smbts_use_search_temp_decay;
            double smbts_search_temp_decay_visits_scale;
            double smdents_entropy_temp_init;
            double smdents_entropy_temp_visits_scale;

            double search_runtime;
            int max_trial_length;
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
                double search_runtime,
                int max_trial_length,
                double eval_delta,
                int rollouts_per_mc_eval,
                int num_repeats,
                int num_threads,
                int eval_threads);

            /**
             * Returns if the env we are using is a python env
            */
            bool is_python_env();

            /**
             * Returns an instance of ThtsEnv to use for this run
            */
            std::shared_ptr<MoThtsEnv> get_env();

            /**
             * Returns and instance of ThtsManager to use for this run
            */
            std::shared_ptr<MoThtsManager> get_thts_manager(std::shared_ptr<MoThtsEnv> env);

            /**
             * Returns a root node to use for search given these params
            */
            std::shared_ptr<MoThtsDNode> get_root_search_node(
                std::shared_ptr<MoThtsEnv> env, std::shared_ptr<MoThtsManager> manager);

            /**
             * The minimum value possible in the environment
             * Useful for setting default values
             * (Note min value for each independent objective, which isn't necessarily achievable)
            */
            Eigen::ArrayXd get_env_min_value();

            /**
             * Returns the max value possible in the environment 
             * (Same as get_env_min_value, but for maximum)
            */
            Eigen::ArrayXd get_env_max_value();

            /**
             * Get max value range (max_value - min_value)
            */
            Eigen::ArrayXd get_env_value_range();
    };

    /**
     * Get a list of run id's from an experiment id
    */
    std::shared_ptr<std::vector<RunID>> get_run_ids_from_expr_id(std::string expr_id);
}