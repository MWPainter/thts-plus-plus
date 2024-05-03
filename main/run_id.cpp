#include "main/run_id.h"

#include "mo/chmcts_manager.h"
#include "mo/czt_manager.h"
#include "mo/smt_bts_manager.h"
#include "mo/smt_dents_manager.h"

#include "mo/czt_decision_node.h"
#include "mo/chmcts_decision_node.h"
#include "mo/smt_bts_decision_node.h"
#include "mo/smt_dents_decision_node.h"

#include "py/pickle_wrapper.h"
#include "py/mo_gym_multiprocessing_thts_env.h"

#include "test/mo/test_mo_thts_env.h"
#include "py/mo_py_multiprocessing_thts_env.h"

#include <stdexcept>

using namespace std;
using namespace thts;
using namespace thts::python;
using namespace thts::test;
using namespace pybind11::literals;

namespace py = pybind11;

namespace thts {
    /**
     * Default constructor
    */
    RunID::RunID() {}

    /**
     * Initialised constructor
    */
    RunID::RunID(
        string env_id,
        string expr_id,
        time_t expr_timestamp,
        string alg_id,
        unordered_map<string, double>& alg_params,
        double search_runtime,
        int max_trial_length,
        double eval_delta,
        int rollouts_per_mc_eval,
        int num_repeats,
        int num_threads,
        int eval_threads) :
            env_id(env_id),
            expr_id(expr_id),
            expr_timestamp(expr_timestamp),
            alg_id(alg_id),
            alg_params(alg_params),
            czt_bias(CztManagerArgs::bias_default),
            czt_ball_split_visit_thresh(BL_MoThtsManagerArgs::num_backups_before_allowed_to_split_default),
            sm_l_inf_thresh(SmtThtsManagerArgs::simplex_node_l_inf_thresh_default),
            sm_max_depth(SmtThtsManagerArgs::simplex_node_max_depth_default),
            sm_split_visit_thresh(SmtThtsManagerArgs::simplex_node_split_visit_thresh_default),
            smbts_search_temp(SmtBtsManagerArgs::temp_default),
            smbts_epsilon(SmtBtsManagerArgs::epsilon_default),
            smbts_use_search_temp_decay(false),
            smbts_search_temp_decay_visits_scale(SmtBtsManagerArgs::temp_decay_visits_scale_default),
            smdents_entropy_temp_init(SmtDentsManagerArgs::value_temp_init_default),
            smdents_entropy_temp_visits_scale(SmtDentsManagerArgs::value_temp_decay_visits_scale_default),
            search_runtime(search_runtime),
            max_trial_length(max_trial_length),
            eval_delta(eval_delta),
            rollouts_per_mc_eval(rollouts_per_mc_eval),
            num_repeats(num_repeats),
            num_threads(num_threads),
            eval_threads(eval_threads),
            num_envs((eval_threads > num_threads) ? eval_threads : num_threads)
    {
        if (alg_params.contains(CZT_BIAS_PARAM_ID)) {
            czt_bias = alg_params[CZT_BIAS_PARAM_ID];
        }
        if (alg_params.contains(CZT_BALL_SPLIT_VISIT_THRESH_PARAM_ID)) {
            czt_ball_split_visit_thresh = alg_params[CZT_BALL_SPLIT_VISIT_THRESH_PARAM_ID];
        }
        if (alg_params.contains(SM_L_INF_THRESH_PARAM_ID)) {
            sm_l_inf_thresh = alg_params[SM_L_INF_THRESH_PARAM_ID];
        }
        if (alg_params.contains(SM_MAX_DEPTH)) {
            sm_max_depth = alg_params[SM_MAX_DEPTH];
        }
        if (alg_params.contains(SM_SPLIT_VISIT_THRESH_PARAM_ID)) {
            sm_split_visit_thresh = alg_params[SM_SPLIT_VISIT_THRESH_PARAM_ID];
        }
        if (alg_params.contains(SMBTS_SEARCH_TEMP_PARAM_ID)) {
            smbts_search_temp = alg_params[SMBTS_SEARCH_TEMP_PARAM_ID];
        }
        if (alg_params.contains(SMBTS_EPSILON_PARAM_ID)) {
            smbts_epsilon = alg_params[SMBTS_EPSILON_PARAM_ID];
        }
        if (alg_params.contains(SMBTS_SEARCH_TEMP_USE_DECAY_PARAM_ID)) {
            smbts_use_search_temp_decay = alg_params[SMBTS_SEARCH_TEMP_USE_DECAY_PARAM_ID];
        }
        if (alg_params.contains(SMBTS_SEARCH_TEMP_DECAY_VISITS_SCALE_PARAM_ID)) {
            smbts_search_temp_decay_visits_scale = alg_params[SMBTS_SEARCH_TEMP_DECAY_VISITS_SCALE_PARAM_ID];
        }
        if (alg_params.contains(SMDENTS_ENTROPY_TEMP_INIT_PARAM_ID)) {
            smdents_entropy_temp_init = alg_params[SMDENTS_ENTROPY_TEMP_INIT_PARAM_ID];
        }
        if (alg_params.contains(SMDENTS_ENTROPY_TEMP_VISITS_SCALE_PARAM_ID)) {
            smdents_entropy_temp_visits_scale = alg_params[SMDENTS_ENTROPY_TEMP_VISITS_SCALE_PARAM_ID];
        }
    }

    bool RunID::is_python_env() 
    {
        return MO_GYM_ENVS.contains(env_id) || DEBUG_PY_ENVS.contains(env_id);
    }

    /**
     * Create and return the env
    */
    shared_ptr<MoThtsEnv> RunID::get_env() 
    {
        if (MO_GYM_ENVS.contains(env_id)) {
            shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
            return make_shared<MoGymMultiprocessingThtsEnv>(pickle_wrapper, env_id);
        }

        if (DEBUG_ENVS.contains(env_id)) {
            int walk_len = 10;
            bool stochastic = (env_id == DEBUG_ENV_2_ID) || (env_id == DEBUG_ENV_4_ID);
            double wrong_dir_prob = stochastic ? 0.25 : 0.0;
            bool add_extra_rewards = (env_id == DEBUG_ENV_3_ID) || (env_id == DEBUG_ENV_4_ID);
            return make_shared<TestMoThtsEnv>(walk_len, wrong_dir_prob, add_extra_rewards);
        }

        if (DEBUG_PY_ENVS.contains(env_id)) {
            int walk_len = 10;
            bool stochastic = (env_id == DEBUG_PY_ENV_2_ID) || (env_id == DEBUG_PY_ENV_4_ID);
            double wrong_dir_prob = stochastic ? 0.25 : 0.0;
            bool add_extra_rewards = (env_id == DEBUG_PY_ENV_3_ID) || (env_id == DEBUG_PY_ENV_4_ID);

            py::module_ py_thts_env_module = py::module_::import("mo_test_env"); 
            py::object py_thts_env_py_obj = py_thts_env_module.attr("MoPyTestThtsEnv")(
                "walk_len"_a=walk_len, "wrong_dir_prob"_a=wrong_dir_prob, "add_extra_rewards"_a=add_extra_rewards);
            shared_ptr<py::object> py_thts_env = make_shared<py::object>(py_thts_env_py_obj);

            shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
            return make_shared<MoPyMultiprocessingThtsEnv>(pickle_wrapper, py_thts_env);
        }

        throw runtime_error("Error in RunID get_env");
    }

    /**
     * Create thts manager
    */
    shared_ptr<MoThtsManager> RunID::get_thts_manager(shared_ptr<MoThtsEnv> env) 
    {
        if (alg_id == CZT_ALG_ID) {
            CztManagerArgs manager_args(env);
            manager_args.max_depth = max_trial_length;
            manager_args.mcts_mode = false;
            manager_args.num_threads = num_threads;
            manager_args.num_envs = num_envs;
            manager_args.bias = czt_bias;
            manager_args.num_backups_before_allowed_to_split = czt_ball_split_visit_thresh;
            // manager_args.use_transposition_table = true;
            return make_shared<CztManager>(manager_args);
        }

        if (alg_id == CHMCTS_ALG_ID) {
            ChmctsManagerArgs manager_args(env);
            manager_args.max_depth = max_trial_length;
            manager_args.mcts_mode = false;
            manager_args.num_threads = num_threads;
            manager_args.num_envs = num_envs;
            manager_args.bias = czt_bias;
            manager_args.num_backups_before_allowed_to_split = czt_ball_split_visit_thresh;
            // manager_args.use_transposition_table = true;
            return make_shared<ChmctsManager>(manager_args);
        }

        if (alg_id == SMBTS_ALG_ID) {
            SmtBtsManagerArgs manager_args(env, get_env_min_value());
            manager_args.max_depth = max_trial_length;
            manager_args.mcts_mode = false;
            manager_args.num_threads = num_threads;
            manager_args.num_envs = num_envs;
            manager_args.simplex_node_l_inf_thresh = sm_l_inf_thresh;
            manager_args.simplex_node_split_visit_thresh = sm_split_visit_thresh;
            manager_args.simplex_node_max_depth = sm_max_depth;
            manager_args.temp = smbts_search_temp;
            manager_args.epsilon = smbts_epsilon;
            manager_args.root_node_epsilon = smbts_epsilon;
            if (smbts_use_search_temp_decay) {
                manager_args.temp_decay_fn = decayed_temp_inv_sqrt;
            }
            manager_args.temp_decay_visits_scale = smbts_search_temp_decay_visits_scale;
            // manager_args.use_transposition_table = true;
            return make_shared<SmtBtsManager>(manager_args);
        }

        if (alg_id == SMDENTS_ALG_ID) {
            SmtDentsManagerArgs manager_args(env, get_env_min_value());
            manager_args.max_depth = max_trial_length;
            manager_args.mcts_mode = false;
            manager_args.num_threads = num_threads;
            manager_args.num_envs = num_envs;
            manager_args.simplex_node_l_inf_thresh = sm_l_inf_thresh;
            manager_args.simplex_node_split_visit_thresh = sm_split_visit_thresh;
            manager_args.simplex_node_max_depth = sm_max_depth;
            manager_args.temp = smbts_search_temp;
            manager_args.epsilon = smbts_epsilon;
            manager_args.root_node_epsilon = smbts_epsilon;
            if (smbts_use_search_temp_decay) {
                manager_args.temp_decay_fn = decayed_temp_inv_sqrt;
            }
            manager_args.temp_decay_visits_scale = smbts_search_temp_decay_visits_scale;
            manager_args.value_temp_init = smdents_entropy_temp_init;
            manager_args.value_temp_decay_visits_scale = smdents_entropy_temp_visits_scale;
            // manager_args.use_transposition_table = true;
            return make_shared<SmtDentsManager>(manager_args);
        }

        throw runtime_error("Error in RunID get_thts_manager");
    }

    /**
     * Return a root search node
    */
    shared_ptr<MoThtsDNode> RunID::get_root_search_node(shared_ptr<MoThtsEnv> env, shared_ptr<MoThtsManager> manager) 
    {
        if (alg_id == CZT_ALG_ID) {
            shared_ptr<CztManager> czt_manager = static_pointer_cast<CztManager>(manager);
            return make_shared<CztDNode>(czt_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == CHMCTS_ALG_ID) {
            shared_ptr<ChmctsManager> chmcts_manager = static_pointer_cast<ChmctsManager>(manager);
            return make_shared<ChmctsDNode>(chmcts_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == SMBTS_ALG_ID) {
            shared_ptr<SmtBtsManager> smbts_manager = static_pointer_cast<SmtBtsManager>(manager);
            return make_shared<SmtBtsDNode>(smbts_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == SMDENTS_ALG_ID) {
            shared_ptr<SmtDentsManager> smdents_manager = static_pointer_cast<SmtDentsManager>(manager);
            return make_shared<SmtDentsDNode>(smdents_manager, env->get_initial_state_itfc(), 0, 0);
        }

        throw runtime_error("Error in RunID get_root_search_node");
    }

    /**
     * Get min value
     * See https://mo-gymnasium.farama.org/ for env defs
    */
    Eigen::ArrayXd RunID::get_env_min_value() 
    {
        if (env_id == DST_ENV_ID) {
            Eigen::ArrayXd min_val = Eigen::ArrayXd(2);
            min_val[0] = 0.0;
            min_val[1] = -1.0 * max_trial_length;
            return min_val;
        }
        if (env_id == FRUIT_TREE_ENV_ID) {
            return Eigen::ArrayXd::Zero(6);
        }


        if (DEBUG_ENVS.contains(env_id) || DEBUG_PY_ENVS.contains(env_id)) {
            unordered_set<string> four_d_envs = 
            {
                DEBUG_ENV_3_ID,
                DEBUG_ENV_4_ID,
                DEBUG_PY_ENV_3_ID,
                DEBUG_PY_ENV_4_ID,
            };
            Eigen::ArrayXd min_val = Eigen::ArrayXd(2);
            if (four_d_envs.contains(env_id)) {
                min_val = Eigen::ArrayXd(4);
            }
            min_val[0] = -10.0;
            min_val[1] = -10.0;
            if (four_d_envs.contains(env_id)) {
                min_val[2] = 0.0;
                min_val[3] = 0.0;
            }
            return min_val;
        }

        throw runtime_error("Error in RunID get_env_min_value");
    }

    /**
     * Get max value
     * See https://mo-gymnasium.farama.org/ for env defs
    */
    Eigen::ArrayXd RunID::get_env_max_value() 
    {
        if (env_id == DST_ENV_ID) {
            Eigen::ArrayXd max_val = Eigen::ArrayXd(2);
            max_val[0] = 23.7;
            max_val[1] = 0.0;
            return max_val;
        }
        if (env_id == FRUIT_TREE_ENV_ID) {
            return Eigen::ArrayXd::Ones(6) * 10.0;
        }
                
        if (DEBUG_ENVS.contains(env_id) || DEBUG_PY_ENVS.contains(env_id)) {
            unordered_set<string> four_d_envs = 
            {
                DEBUG_ENV_3_ID,
                DEBUG_ENV_4_ID,
                DEBUG_PY_ENV_3_ID,
                DEBUG_PY_ENV_4_ID,
            };
            Eigen::ArrayXd max_val = Eigen::ArrayXd(2);
            if (four_d_envs.contains(env_id)) {
                max_val = Eigen::ArrayXd(4);
            }
            max_val[0] = -5.0;
            max_val[1] = -5.0;
            if (four_d_envs.contains(env_id)) {
                max_val[2] = 2.0;
                max_val[3] = 2.0;
            }
            return max_val;
        }

        throw runtime_error("Error in RunID get_env_max_value");
    }

    /**
     * Get value range
    */
    Eigen::ArrayXd RunID::get_env_value_range() 
    {
        return get_env_max_value() - get_env_min_value();
    }

    /**
     * Gets a list of RunID objects from a given expr id
    */
    shared_ptr<vector<RunID>> get_run_ids_from_expr_id(string expr_id) 
    {
        shared_ptr<vector<RunID>> run_ids = make_shared<vector<RunID>>();

        // expr_id: 000_debug 
        // debug expr id for debugging
        if (expr_id == DEBUG_EXPR_ID) {
            string env_id = DST_ENV_ID;
            // string env_id = DEBUG_PY_ENV_1_ID;
            time_t expr_timestamp = std::time(nullptr);
            double search_runtime = 5.0;
            int max_trial_length = 50;
            double eval_delta = 1.0;
            int rollouts_per_mc_eval = 250;
            int num_repeats = 3;
            int num_threads = 1;
            int eval_threads = 1;

            unordered_map<string,double> alg_params =
            {
                {CZT_BIAS_PARAM_ID, 4.0},
                {CZT_BALL_SPLIT_VISIT_THRESH_PARAM_ID, 10.0},
                {SM_L_INF_THRESH_PARAM_ID, 0.05},
                // {SM_MAX_DEPTH, 10.0},
                // {SM_SPLIT_VISIT_THRESH_PARAM_ID, 10.0},
                {SM_SPLIT_VISIT_THRESH_PARAM_ID, 1.0},
                {SMBTS_SEARCH_TEMP_PARAM_ID, 100.0},
                {SMBTS_EPSILON_PARAM_ID, 0.1},
                // {SMBTS_SEARCH_TEMP_USE_DECAY_PARAM_ID, 1.0},
                {SMBTS_SEARCH_TEMP_USE_DECAY_PARAM_ID, 0.0},
                {SMBTS_SEARCH_TEMP_DECAY_VISITS_SCALE_PARAM_ID, 1.0},
                {SMDENTS_ENTROPY_TEMP_INIT_PARAM_ID, 0.5},
                {SMDENTS_ENTROPY_TEMP_VISITS_SCALE_PARAM_ID, 1.0},
            };

            vector<string> alg_ids = 
            {
                SMBTS_ALG_ID,
                // SMDENTS_ALG_ID,
                // CZT_ALG_ID,
                // CHMCTS_ALG_ID,
            };

            for (string alg_id : alg_ids) {
                run_ids->push_back(RunID(
                    env_id,
                    expr_id,
                    expr_timestamp,
                    alg_id,
                    alg_params,
                    search_runtime,
                    max_trial_length,
                    eval_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads
                ));
            }

            return run_ids;
        }

        // expr_id: 001_debug ... 008_debug
        // debug expr ids for running Python vs C++ environment testing
        unordered_map<string,string> py_vs_cpp_debug_env_ids = 
        {
            {DEBUG_ENV_1_EXPR_ID, DEBUG_ENV_1_ID},
            {DEBUG_ENV_2_EXPR_ID, DEBUG_ENV_2_ID},
            {DEBUG_ENV_3_EXPR_ID, DEBUG_ENV_3_ID},
            {DEBUG_ENV_4_EXPR_ID, DEBUG_ENV_4_ID},
            {DEBUG_PY_ENV_1_EXPR_ID, DEBUG_PY_ENV_1_ID},
            {DEBUG_PY_ENV_2_EXPR_ID, DEBUG_PY_ENV_2_ID},
            {DEBUG_PY_ENV_3_EXPR_ID, DEBUG_PY_ENV_3_ID},
            {DEBUG_PY_ENV_4_EXPR_ID, DEBUG_PY_ENV_4_ID},
        }; 
        if (py_vs_cpp_debug_env_ids.contains(expr_id)) {
            string env_id = py_vs_cpp_debug_env_ids[expr_id];
            time_t expr_timestamp = std::time(nullptr);
            double search_runtime = 15.0;
            int max_trial_length = 50;
            double eval_delta = 1.0;
            int rollouts_per_mc_eval = 250;
            int num_repeats = 3;
            int num_threads = 16;
            int eval_threads = 16;

            unordered_map<string,double> alg_params =
            {
                {CZT_BIAS_PARAM_ID, 100.0},
                {CZT_BALL_SPLIT_VISIT_THRESH_PARAM_ID, 10.0},
                {SM_L_INF_THRESH_PARAM_ID, 0.05},
                // {SM_MAX_DEPTH, 10.0},
                // {SM_SPLIT_VISIT_THRESH_PARAM_ID, 10.0},
                {SM_SPLIT_VISIT_THRESH_PARAM_ID, 1.0},
                {SMBTS_SEARCH_TEMP_PARAM_ID, 1.0},
                {SMBTS_EPSILON_PARAM_ID, 0.1},
                // {SMBTS_SEARCH_TEMP_USE_DECAY_PARAM_ID, 1.0},
                {SMBTS_SEARCH_TEMP_USE_DECAY_PARAM_ID, 0.0},
                {SMBTS_SEARCH_TEMP_DECAY_VISITS_SCALE_PARAM_ID, 1.0},
                {SMDENTS_ENTROPY_TEMP_INIT_PARAM_ID, 0.1},
                {SMDENTS_ENTROPY_TEMP_VISITS_SCALE_PARAM_ID, 1.0},
            };

            vector<string> alg_ids = 
            {
                SMBTS_ALG_ID,
                SMDENTS_ALG_ID,
                CZT_ALG_ID,
                CHMCTS_ALG_ID,
            };

            for (string alg_id : alg_ids) {
                run_ids->push_back(RunID(
                    env_id,
                    expr_id,
                    expr_timestamp,
                    alg_id,
                    alg_params,
                    search_runtime,
                    max_trial_length,
                    eval_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads
                ));
            }

            return run_ids;
        }

        // expr_id: 009_poc_dst
        // proof of concept with deep sea treasure
        if (expr_id == POC_DST_EXPR_ID) {
            string env_id = DST_ENV_ID;
            time_t expr_timestamp = std::time(nullptr);
            double search_runtime = 30.0;
            int max_trial_length = 50;
            double eval_delta = 1.0;
            int rollouts_per_mc_eval = 1000;
            int num_repeats = 3;
            int num_threads = 16;
            int eval_threads = 16;

            unordered_map<string,double> alg_params =
            {
                {CZT_BIAS_PARAM_ID, 4.0},
                {CZT_BALL_SPLIT_VISIT_THRESH_PARAM_ID, 10.0},
                {SM_L_INF_THRESH_PARAM_ID, 0.05},
                // {SM_MAX_DEPTH, 10.0},
                {SM_SPLIT_VISIT_THRESH_PARAM_ID, 10.0},
                // {SMBTS_SEARCH_TEMP_PARAM_ID, 1.0},
                {SMBTS_SEARCH_TEMP_PARAM_ID, 10.0},
                {SMBTS_EPSILON_PARAM_ID, 0.01},
                // {SMBTS_SEARCH_TEMP_USE_DECAY_PARAM_ID, 1.0},
                {SMBTS_SEARCH_TEMP_USE_DECAY_PARAM_ID, 0.0},
                {SMBTS_SEARCH_TEMP_DECAY_VISITS_SCALE_PARAM_ID, 1.0},
                {SMDENTS_ENTROPY_TEMP_INIT_PARAM_ID, 1.0},
                {SMDENTS_ENTROPY_TEMP_VISITS_SCALE_PARAM_ID, 1.0},
            };

            vector<string> alg_ids = 
            {
                SMDENTS_ALG_ID,
                SMBTS_ALG_ID,
                CZT_ALG_ID,
                CHMCTS_ALG_ID,
            };

            for (string alg_id : alg_ids) {
                run_ids->push_back(RunID(
                    env_id,
                    expr_id,
                    expr_timestamp,
                    alg_id,
                    alg_params,
                    search_runtime,
                    max_trial_length,
                    eval_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads
                ));
            }

            return run_ids;
        }

        // expr_id: 010_poc_dst
        // proof of concept with deep sea treasure
        if (expr_id == POC_FT_EXPR_ID) {
            string env_id = FRUIT_TREE_ENV_ID;
            time_t expr_timestamp = std::time(nullptr);
            double search_runtime = 10.0;
            int max_trial_length = 50;
            double eval_delta = 0.5;
            int rollouts_per_mc_eval = 1000;
            int num_repeats = 3;
            int num_threads = 16;
            int eval_threads = 16;

            unordered_map<string,double> alg_params =
            {
                {CZT_BIAS_PARAM_ID, 4.0},
                {CZT_BALL_SPLIT_VISIT_THRESH_PARAM_ID, 10.0},
                {SM_L_INF_THRESH_PARAM_ID, 0.05},
                // {SM_MAX_DEPTH, 10.0},
                {SM_SPLIT_VISIT_THRESH_PARAM_ID, 10.0},
                // {SMBTS_SEARCH_TEMP_PARAM_ID, 1.0},
                {SMBTS_SEARCH_TEMP_PARAM_ID, 10.0},
                {SMBTS_EPSILON_PARAM_ID, 0.01},
                // {SMBTS_SEARCH_TEMP_USE_DECAY_PARAM_ID, 1.0},
                {SMBTS_SEARCH_TEMP_USE_DECAY_PARAM_ID, 0.0},
                {SMBTS_SEARCH_TEMP_DECAY_VISITS_SCALE_PARAM_ID, 1.0},
                {SMDENTS_ENTROPY_TEMP_INIT_PARAM_ID, 1.0},
                {SMDENTS_ENTROPY_TEMP_VISITS_SCALE_PARAM_ID, 1.0},
            };

            vector<string> alg_ids = 
            {
                SMDENTS_ALG_ID,
                SMBTS_ALG_ID,
                CZT_ALG_ID,
                CHMCTS_ALG_ID,
            };

            for (string alg_id : alg_ids) {
                run_ids->push_back(RunID(
                    env_id,
                    expr_id,
                    expr_timestamp,
                    alg_id,
                    alg_params,
                    search_runtime,
                    max_trial_length,
                    eval_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads
                ));
            }

            return run_ids;
        }

        throw runtime_error("Error in get_run_ids_from_expr_id");
    }

}