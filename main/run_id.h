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
static const std::string DST_ENV_ID = "deep-sea-treasure-v0";                   // 50x + 70x
static const std::string DST_CONC_ENV_ID = "deep-sea-treasure-concave-v0";      // 51x + 71x
static const std::string DST_MIRR_ENV_ID = "deep-sea-treasure-mirrored-v0";     // 52x + 72x
static const std::string RESOURCE_GATHER_ENV_ID = "resource-gathering-v0";      // 53x + 73x
static const std::string BREAKABLE_BOTTLES_ENV_ID = "breakable-bottles-v0";     // 55x + 75x
static const std::string FRUIT_TREE_ENV_ID = "fruit-tree-v0";                   // 56x + 76x
static const std::string FOUR_ROOM_ENV_ID = "four-room-v0";                     // 57x + 77x

// env ids - mo gymnasium (cts obs / discr act)
static const std::string MOUNTAIN_CAR_ENV_ID = "mo-mountaincar-v0";             // 59x + 79x
static const std::string LUNAR_LANDER_ENV_ID = "mo-lunar-lander-v2";            // 60x + 80x
static const std::string MINECART_ENV_ID = "minecart-v0";                       // 61x + 81x
static const std::string HIGHWAY_ENV_ID = "mo-highway-v0";                      // 62x + 82x
static const std::string HIGHWAY_FAST_ENV_ID = "mo-highway-fast-v0";            // 63x + 83x

// env ids - mo gymnasium (with extra time cost)
static const std::string RESOURCE_GATHER_TIMED_ENV_ID = "resource-gathering-timed-v0";  // 54x + 74x
static const std::string FOUR_ROOM_TIMED_ENV_ID = "four-room-timed-v0";                 // 58x + 78x
static const std::unordered_map<std::string,std::string> TIMED_ENV_ID_TO_GYM_ID =
{
    {RESOURCE_GATHER_TIMED_ENV_ID, RESOURCE_GATHER_ENV_ID},
    {FOUR_ROOM_TIMED_ENV_ID, FOUR_ROOM_ENV_ID},
};

// lists of envs - debug envs
static const std::unordered_set<std::string> DEBUG_ENVS =
{
     DEBUG_ENV_1_ID,
     DEBUG_ENV_2_ID,
     DEBUG_ENV_3_ID,
     DEBUG_ENV_4_ID,
};

// lists of envs - debug py envs
static const std::unordered_set<std::string> DEBUG_PY_ENVS =
{
     DEBUG_PY_ENV_1_ID,
     DEBUG_PY_ENV_2_ID,
     DEBUG_PY_ENV_3_ID,
     DEBUG_PY_ENV_4_ID,
};

// lists of envs - toy envas
static const std::unordered_set<std::string> TOY_ENVS =
{
};

// lists of envs - mo gymnasium envs
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

// expr ids - testing - for VS debugging
static const std::string DEBUG_EXPR_ID = "000_debug";

// expr ids - testing - debugging 
static const std::string DEBUG_ENV_1_EXPR_ID = "001_debug_env_1";
static const std::string DEBUG_PY_ENV_1_EXPR_ID = "002_debug_py_env_1";
static const std::string DEBUG_ENV_2_EXPR_ID = "003_debug_env_2";
static const std::string DEBUG_PY_ENV_2_EXPR_ID = "004_debug_py_env_2";
static const std::string DEBUG_ENV_3_EXPR_ID = "005_debug_env_3";
static const std::string DEBUG_PY_ENV_3_EXPR_ID = "006_debug_py_env_3";
static const std::string DEBUG_ENV_4_EXPR_ID = "007_debug_env_4";
static const std::string DEBUG_PY_ENV_4_EXPR_ID = "008_debug_py_env_4";

// expr ids - testing - proof of concept tests on mo-gym envs (running without hyperparam tuning)
static const std::string POC_DST_EXPR_ID = "009_poc_dst";
static const std::string POC_FT_EXPR_ID = "010_poc_ft";

// expr ids - testing - debugging HP opt
static const std::string DEBUG_CZT_HP_OPT_EXPR_ID = "020_debug_czt_hp";
static const std::string DEBUG_CHMCTS_HP_OPT_EXPR_ID = "021_debug_chmcts_hp";
static const std::string DEBUG_SMBTS_HP_OPT_EXPR_ID = "022_debug_smbts_hp";
static const std::string DEBUG_SMDENTS_HP_OPT_EXPR_ID = "023_debug_smdents_hp";

// expr ids - toy/tree (1xx + 2xx = hyperparam tuning)
// TODO


// expr ids - toy/tree (3xx + 4xx = eval)
// TODO

// expr ids - mo gymnasium (5xx + 6xx = hyperparam tuning)
// - deep sea treasure
static const std::string HP_OPT_DST_CZT_EXPR_ID = "500_hp_opt_dst_czt";
static const std::string HP_OPT_DST_CHMCTS_EXPR_ID = "501_hp_opt_dst_chmcts";
static const std::string HP_OPT_DST_SMBTS_EXPR_ID = "502_hp_opt_dst_smbts";
static const std::string HP_OPT_DST_SMDENTS_EXPR_ID = "503_hp_opt_dst_smdents";
// - breakable bottles
static const std::string HP_OPT_BB_CZT_EXPR_ID = "550_hp_opt_bb_czt";
static const std::string HP_OPT_BB_CHMCTS_EXPR_ID = "551_hp_opt_bb_chmcts";
static const std::string HP_OPT_BB_SMBTS_EXPR_ID = "552_hp_opt_bb_smbts";
static const std::string HP_OPT_BB_SMDENTS_EXPR_ID = "553_hp_opt_bb_smdents";
// - fruit tree
static const std::string HP_OPT_FT_CZT_EXPR_ID = "560_hp_opt_ft_czt";
static const std::string HP_OPT_FT_CHMCTS_EXPR_ID = "561_hp_opt_ft_chmcts";
static const std::string HP_OPT_FT_SMBTS_EXPR_ID = "562_hp_opt_ft_smbts";
static const std::string HP_OPT_FT_SMDENTS_EXPR_ID = "563_hp_opt_ft_smdents";
// - four room (timed)
static const std::string HP_OPT_FOUR_T_CZT_EXPR_ID = "580_hp_opt_four_czt";
static const std::string HP_OPT_FOUR_T_CHMCTS_EXPR_ID = "581_hp_opt_four_chmcts";
static const std::string HP_OPT_FOUR_T_SMBTS_EXPR_ID = "582_hp_opt_four_smbts";
static const std::string HP_OPT_FOUR_T_SMDENTS_EXPR_ID = "583_hp_opt_four_smdents";
// - minecart
static const std::string HP_OPT_MINE_CZT_EXPR_ID = "610_hp_opt_mine_czt";
static const std::string HP_OPT_MINE_CHMCTS_EXPR_ID = "611_hp_opt_mine_chmcts";
static const std::string HP_OPT_MINE_SMBTS_EXPR_ID = "612_hp_opt_mine_smbts";
static const std::string HP_OPT_MINE_SMDENTS_EXPR_ID = "613_hp_opt_mine_smdents";

// expr ids - lists of czt / chmcts / bts / dents expr_ids
static std::unordered_map<std::string,std::string> HP_OPT_MOGYM_CZT_EXPR_ID_TO_ENV_ID =
{
    {HP_OPT_DST_CZT_EXPR_ID,        DST_ENV_ID},
    {HP_OPT_BB_CZT_EXPR_ID,         BREAKABLE_BOTTLES_ENV_ID},
    {HP_OPT_FT_CZT_EXPR_ID,         FRUIT_TREE_ENV_ID},
    {HP_OPT_FOUR_T_CZT_EXPR_ID,     FOUR_ROOM_TIMED_ENV_ID},
    {HP_OPT_MINE_CZT_EXPR_ID,       MINECART_ENV_ID},
};
static std::unordered_map<std::string,std::string> HP_OPT_MOGYM_CHMCTS_EXPR_ID_TO_ENV_ID =
{
    {HP_OPT_DST_CHMCTS_EXPR_ID,     DST_ENV_ID},
    {HP_OPT_BB_CHMCTS_EXPR_ID,      BREAKABLE_BOTTLES_ENV_ID},
    {HP_OPT_FT_CHMCTS_EXPR_ID,      FRUIT_TREE_ENV_ID},
    {HP_OPT_FOUR_T_CHMCTS_EXPR_ID,  FOUR_ROOM_TIMED_ENV_ID},
    {HP_OPT_MINE_CHMCTS_EXPR_ID,    MINECART_ENV_ID},
};
static std::unordered_map<std::string,std::string> HP_OPT_MOGYM_SMBTS_EXPR_ID_TO_ENV_ID =
{   
    {HP_OPT_DST_SMBTS_EXPR_ID,      DST_ENV_ID},
    {HP_OPT_BB_SMBTS_EXPR_ID,       BREAKABLE_BOTTLES_ENV_ID},
    {HP_OPT_FT_SMBTS_EXPR_ID,       FRUIT_TREE_ENV_ID},
    {HP_OPT_FOUR_T_SMBTS_EXPR_ID,   FOUR_ROOM_TIMED_ENV_ID},
    {HP_OPT_MINE_SMBTS_EXPR_ID,     MINECART_ENV_ID},
};
static std::unordered_map<std::string,std::string> HP_OPT_MOGYM_SMDENTS_EXPR_ID_TO_ENV_ID =
{
    {HP_OPT_DST_SMDENTS_EXPR_ID,        DST_ENV_ID},
    {HP_OPT_BB_SMDENTS_EXPR_ID,         BREAKABLE_BOTTLES_ENV_ID},
    {HP_OPT_FT_SMDENTS_EXPR_ID,         FRUIT_TREE_ENV_ID},
    {HP_OPT_FOUR_T_SMDENTS_EXPR_ID,     FOUR_ROOM_TIMED_ENV_ID},
    {HP_OPT_MINE_SMDENTS_EXPR_ID,       MINECART_ENV_ID},
};

// expr ids - mo gymnasium (7xx + 8xx = eval)

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

// relevant alg ids -> param ids
static std::unordered_map<std::string,std::vector<std::string>> RELEVANT_PARAM_IDS =
{
    {CZT_ALG_ID,
        {
            CZT_BIAS_PARAM_ID,
            CZT_BALL_SPLIT_VISIT_THRESH_PARAM_ID,
        },
    },
    {CHMCTS_ALG_ID,
        {
            CZT_BIAS_PARAM_ID,
            CZT_BALL_SPLIT_VISIT_THRESH_PARAM_ID,
        },
    },
    {SMBTS_ALG_ID,
        {
            SM_L_INF_THRESH_PARAM_ID,
            // SM_MAX_DEPTH,
            SM_SPLIT_VISIT_THRESH_PARAM_ID,
            SMBTS_SEARCH_TEMP_PARAM_ID,
            SMBTS_EPSILON_PARAM_ID,
            SMBTS_SEARCH_TEMP_USE_DECAY_PARAM_ID,
            SMBTS_SEARCH_TEMP_DECAY_VISITS_SCALE_PARAM_ID,
        },
    },
    {SMDENTS_ALG_ID,
        {
            SM_L_INF_THRESH_PARAM_ID,
            // SM_MAX_DEPTH,
            SM_SPLIT_VISIT_THRESH_PARAM_ID,
            SMBTS_SEARCH_TEMP_PARAM_ID,
            SMBTS_EPSILON_PARAM_ID,
            SMBTS_SEARCH_TEMP_USE_DECAY_PARAM_ID,
            SMBTS_SEARCH_TEMP_DECAY_VISITS_SCALE_PARAM_ID,
            SMDENTS_ENTROPY_TEMP_INIT_PARAM_ID,
            SMDENTS_ENTROPY_TEMP_VISITS_SCALE_PARAM_ID
        },
    },
};

// List of boolean + int param ids
static std::unordered_set<std::string> BOOLEAN_PARAM_IDS =
{
    SMBTS_SEARCH_TEMP_USE_DECAY_PARAM_ID,
};

static std::unordered_set<std::string> INTEGER_PARAM_IDS =
{
    CZT_BALL_SPLIT_VISIT_THRESH_PARAM_ID,
    // SM_MAX_DEPTH,
    SM_SPLIT_VISIT_THRESH_PARAM_ID,
};


namespace thts {
    /**
     * Create the env corresponding to 'env_id' and return is
     */
    std::shared_ptr<MoThtsEnv> get_env(std::string env_id);

    /**
     * Checks if env corresponding to 'env_id' is a python env
     */
    bool is_python_env(std::string env_id);

    /**
     * The minimum value possible in the environment
     * Useful for setting default values
     * (Note min value for each independent objective, which isn't necessarily achievable)
    */
    Eigen::ArrayXd get_env_min_value(std::string env_id, int max_trial_length);

    /**
     * Returns the max value possible in the environment 
     * (Same as get_env_min_value, but for maximum)
    */
    Eigen::ArrayXd get_env_max_value(std::string env_id);

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
}