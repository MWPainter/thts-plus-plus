#include "main/run_id.h"

#include "main/run_expr.h"

#include "algorithms/uct/uct_manager.h"
#include "algorithms/uct/hmcts_manager.h"
#include "algorithms/ments/ments_manager.h"
#include "algorithms/ments/dents/dents_manager.h"

#include "algorithms/uct/uct_decision_node.h"
#include "algorithms/ments/ments_decision_node.h"
#include "algorithms/est/est_decision_node.h"
#include "algorithms/ments/dents/dents_decision_node.h"
#include "algorithms/uct/hmcts_decision_node.h"
#include "algorithms/uct/max_uct_decision_node.h"
#include "algorithms/ments/rents/rents_decision_node.h"
#include "algorithms/ments/tents/tents_decision_node.h"

#include "algorithms/common/decaying_temp.h"

#include "py/pickle_wrapper.h"
#include "py/py_multiprocessing_thts_env.h"
#include "py/gym_multiprocessing_thts_env.h"

#include "main/envs/d_chain.h"
#include "main/envs/entropy_trap.h"
#include "main/envs/frozen_lake.h"
#include "main/envs/sailing.h"

#include <cmath>

#include <sstream>
#include <stdexcept>

using namespace std;
using namespace thts;
using namespace thts::python;

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
        bool eval_wrt_time,
        double search_runtime,
        double eval_delta,
        int rollouts_per_mc_eval,
        int max_trial_length,
        int num_repeats,
        int num_threads,
        int eval_threads) :
            env_id(env_id),
            expr_id(expr_id),
            expr_timestamp(expr_timestamp),
            alg_id(alg_id),
            alg_params(alg_params),
            adaptive_bias(UctManagerArgs::adaptive_bias_default),
            bias(UctManagerArgs::bias_default),
            hmcts_uct_budget(HmctsManagerArgs::uct_budget_threshold_default),
            normalise_q_values(MentsManagerArgs::normalise_q_values_default),
            temp(MentsManagerArgs::temp_default),
            decay_fn(DECAY_FN_CONST),
            decay_fn_scale(1.0),
            entropy_coeff(1.0),
            entropy_decay_fn(DECAY_FN_CONST),
            entropy_decay_fn_scale(1.0),
            eval_wrt_time(eval_wrt_time),
            search_runtime(search_runtime),
            eval_delta(eval_delta),
            rollouts_per_mc_eval(rollouts_per_mc_eval),
            max_trial_length(max_trial_length),
            num_repeats(num_repeats),
            num_threads(num_threads),
            eval_threads(eval_threads),
            num_envs((eval_threads > num_threads) ? eval_threads : num_threads)
    {
        if (alg_params.contains(ADAPTIVE_BIAS_PARAM_ID)) {
            adaptive_bias = alg_params[ADAPTIVE_BIAS_PARAM_ID];
        }
        if (alg_params.contains(BIAS_PARAM_ID)) {
            bias = alg_params[BIAS_PARAM_ID];
        }
        if (alg_params.contains(UCT_BUDGET_PARAM_ID)) {
            hmcts_uct_budget = alg_params[UCT_BUDGET_PARAM_ID];
        }
        if (alg_params.contains(NORMALISE_Q_VALUES_PARAM_ID)) {
            normalise_q_values = alg_params[NORMALISE_Q_VALUES_PARAM_ID];
        }
        if (alg_params.contains(TEMP_PARAM_ID)) {
            temp = alg_params[TEMP_PARAM_ID];
        }
        if (alg_params.contains(DECAY_FN_PARAM_ID)) {
            decay_fn = alg_params[DECAY_FN_PARAM_ID];
        }
        if (alg_params.contains(DECAY_FN_SCALE_PARAM_ID)) {
            decay_fn_scale = alg_params[DECAY_FN_SCALE_PARAM_ID];
        }
        if (alg_params.contains(ENTROPY_COEFF_PARAM_ID)) {
            entropy_coeff = alg_params[ENTROPY_COEFF_PARAM_ID];
        }
        if (alg_params.contains(ENTROPY_DECAY_FN_PARAM_ID)) {
            entropy_decay_fn = alg_params[ENTROPY_DECAY_FN_PARAM_ID];
        }
        if (alg_params.contains(ENTROPY_DECAY_FN_SCALE_PARAM_ID)) {
            entropy_decay_fn_scale = alg_params[ENTROPY_DECAY_FN_SCALE_PARAM_ID];
        }
    }

    string RunID::get_results_dir() 
    {
        return thts::get_results_dir(*this);
    }

    bool RunID::is_python_env() 
    {
        return thts::is_python_env(env_id);
    }

    shared_ptr<ThtsEnv> RunID::get_env() 
    {
        return thts::get_env(*this);
    }

    /**
     * Create thts manager
    */
    shared_ptr<ThtsManager> RunID::get_thts_manager(shared_ptr<ThtsEnv> env) 
    {
        if (alg_id == UCT_ALG_ID || alg_id == MAX_UCT_ALG_ID) {
            UctManagerArgs manager_args(env);
            manager_args.max_depth = max_trial_length;
            manager_args.mcts_mode = false;
            manager_args.num_threads = num_threads;
            manager_args.num_envs = num_envs;
            manager_args.adaptive_bias = adaptive_bias;
            manager_args.bias = bias;
            return make_shared<UctManager>(manager_args);
        }

        if (alg_id == MENTS_ALG_ID || alg_id == RENTS_ALG_ID || alg_id == TENTS_ALG_ID) {
            MentsManagerArgs manager_args(env);
            manager_args.max_depth = max_trial_length;
            manager_args.mcts_mode = false;
            manager_args.num_threads = num_threads;
            manager_args.num_envs = num_envs;
            manager_args.normalise_q_values = normalise_q_values;
            manager_args.temp = temp;
            if (alg_params.contains(DEFAULT_Q_VALUE_PARAM_ID)) {
                manager_args.default_q_value = alg_params.at(DEFAULT_Q_VALUE_PARAM_ID);
            }
            return make_shared<MentsManager>(manager_args);
        }

        if (alg_id == BTS_ALG_ID) {
            DentsManagerArgs manager_args(env);
            manager_args.max_depth = max_trial_length;
            manager_args.mcts_mode = false;
            manager_args.num_threads = num_threads;
            manager_args.num_envs = num_envs;
            manager_args.normalise_q_values = normalise_q_values;

            if (alg_params.contains(DEFAULT_Q_VALUE_PARAM_ID)) {
                manager_args.default_q_value = alg_params.at(DEFAULT_Q_VALUE_PARAM_ID);
            }

            // alpha
            manager_args.temp = temp;
            manager_args.temp_decay_fn = nullptr; 
            if (decay_fn == DECAY_FN_INV_SQRT) {
                manager_args.temp_decay_fn = decayed_temp_inv_sqrt;
            } else if (decay_fn == DECAY_FN_INV_LOG) {
                manager_args.temp_decay_fn = decayed_temp_inv_log;
            }
            manager_args.temp_decay_visits_scale = decay_fn_scale;
            
            return make_shared<DentsManager>(manager_args);
        }

        if (alg_id == DENTS_ALG_ID) {
            DentsManagerArgs manager_args(env);
            manager_args.max_depth = max_trial_length;
            manager_args.mcts_mode = false;
            manager_args.num_threads = num_threads;
            manager_args.num_envs = num_envs;
            manager_args.normalise_q_values = normalise_q_values;

            if (alg_params.contains(DEFAULT_Q_VALUE_PARAM_ID)) {
                manager_args.default_q_value = alg_params.at(DEFAULT_Q_VALUE_PARAM_ID);
            }

            // alpha
            manager_args.temp = temp;
            manager_args.temp_decay_fn = nullptr; 
            if (decay_fn == DECAY_FN_INV_SQRT) {
                manager_args.temp_decay_fn = decayed_temp_inv_sqrt;
            } else if (decay_fn == DECAY_FN_INV_LOG) {
                manager_args.temp_decay_fn = decayed_temp_inv_log;
            }
            manager_args.temp_decay_visits_scale = decay_fn_scale;

            // beta
            manager_args.value_temp_init = entropy_coeff;
            manager_args.value_temp_decay_fn = nullptr;
            if (entropy_decay_fn == DECAY_FN_INV_SQRT) {
                manager_args.value_temp_decay_fn = decayed_temp_inv_sqrt;
            } else if (entropy_decay_fn == DECAY_FN_INV_LOG) {
                manager_args.value_temp_decay_fn = decayed_temp_inv_log;
            }
            manager_args.value_temp_decay_visits_scale = entropy_decay_fn_scale;
            
            return make_shared<DentsManager>(manager_args);
        }

        if (alg_id == HMCTS_ALG_ID) {
            HmctsManagerArgs manager_args(env);
            manager_args.max_depth = max_trial_length;
            manager_args.mcts_mode = false;
            manager_args.num_threads = num_threads;
            manager_args.num_envs = num_envs;
            manager_args.adaptive_bias = adaptive_bias;
            manager_args.bias = bias;

            manager_args.total_budget = search_runtime; // search_runtime in units of #trials
            manager_args.uct_budget_threshold = hmcts_uct_budget;

            return make_shared<HmctsManager>(manager_args);
        }

        stringstream ss;
        ss << "Error in RunID get_thts_manager for alg_id = " << alg_id;
        throw runtime_error(ss.str());
    }

    /**
     * Return a root search node
    */
    shared_ptr<ThtsDNode> RunID::get_root_search_node(shared_ptr<ThtsEnv> env, shared_ptr<ThtsManager> manager) 
    {
        if (alg_id == UCT_ALG_ID) {
            shared_ptr<UctManager> uct_manager = static_pointer_cast<UctManager>(manager);
            return make_shared<UctDNode>(uct_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == MENTS_ALG_ID) {
            shared_ptr<MentsManager> ments_manager = static_pointer_cast<DentsManager>(manager);
            return make_shared<MentsDNode>(ments_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == BTS_ALG_ID) {
            shared_ptr<DentsManager> bts_manager = static_pointer_cast<DentsManager>(manager);
            return make_shared<EstDNode>(bts_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == DENTS_ALG_ID) {
            shared_ptr<DentsManager> dents_manager = static_pointer_cast<DentsManager>(manager);
            return make_shared<DentsDNode>(dents_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == RENTS_ALG_ID) {
            shared_ptr<MentsManager> ments_manager = static_pointer_cast<MentsManager>(manager);
            return make_shared<RentsDNode>(ments_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == TENTS_ALG_ID) {
            shared_ptr<MentsManager> ments_manager = static_pointer_cast<MentsManager>(manager);
            return make_shared<TentsDNode>(ments_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == HMCTS_ALG_ID) {
            shared_ptr<HmctsManager> uct_manager = static_pointer_cast<HmctsManager>(manager);
            return make_shared<UctDNode>(uct_manager, env->get_initial_state_itfc(), 0, 0);
        }

        stringstream ss;
        ss << "Error in RunID get_root_search_node for alg_id = " << alg_id;
        throw runtime_error(ss.str());
    }

    /**
     * Gets a list of RunID objects from a given expr id
    */
    shared_ptr<vector<RunID>> get_run_ids_from_expr_id_prefix(string expr_id_prefix) 
    {   
        string expr_id = lookup_expr_id_from_prefix(expr_id_prefix);
        shared_ptr<vector<RunID>> run_ids = make_shared<vector<RunID>>();

        // TODO: define RunId's for experiments

        // expr_id: 000_debug 
        // debug expr id for debugging
        if (expr_id == DEBUG_EXPR_ID) {
            string env_id = D_CHAIN_10_ENV_ID;
            int max_trial_length = ENV_ID_MAX_TRIAL_LEN.at(D_CHAIN_10_ENV_ID);
            time_t expr_timestamp = std::time(nullptr);
            bool eval_wrt_time = false;
            double search_runtime = 100;
            double eval_delta = 25;
            int rollouts_per_mc_eval = 5;
            int num_repeats = 2;
            int num_threads = 1;
            int eval_threads = 1;

            unordered_map<string,double> alg_params =
            {
                {BIAS_PARAM_ID, 4.0},
                {TEMP_PARAM_ID, 1.0},
                {DECAY_FN_PARAM_ID, DECAY_FN_CONST},
                {DECAY_FN_SCALE_PARAM_ID, 1.0},
                {ENTROPY_COEFF_PARAM_ID, 1.0},
                {ENTROPY_DECAY_FN_PARAM_ID, DECAY_FN_CONST},
                {ENTROPY_DECAY_FN_SCALE_PARAM_ID, 1.0},
                {EPSILON_PARAM_ID, 0.1},
            };

            vector<string> alg_ids = 
            {
                UCT_ALG_ID,
                MENTS_ALG_ID,
                BTS_ALG_ID,
                DENTS_ALG_ID,
            };

            for (string alg_id : alg_ids) {
                run_ids->push_back(RunID(
                    env_id,
                    expr_id,
                    expr_timestamp,
                    alg_id,
                    alg_params,
                    eval_wrt_time,
                    search_runtime,
                    eval_delta,
                    rollouts_per_mc_eval,
                    max_trial_length,
                    num_repeats,
                    num_threads,
                    eval_threads
                ));
            }

            return run_ids;
        }

        // ----
        // expr_id: 100_supp_dchain_temp_vary 
        // 10-chain vs temp param
        // ----
        // expr_id: 101_supp_mod_dchain_temp_vary 
        // modified 10-chain vs temp param
        // ----
        // expr_id: 102_supp_entropy_temp_vary 
        // entropy trap 10 vs temp param
        // ----
        // expr_id: 103_supp_entropy_temp_15_vary 
        // entropy trap 15 vs temp param
        // ----
        if (expr_id == SUPP_100_DCHAIN_10_TEMP_EXPR_ID
            || expr_id == SUPP_101_MOD_DCHAIN_10_TEMP_EXPR_ID
            || expr_id == SUPP_102_ENTROPY_TRAP_10_TEMP_EXPR_ID
            || expr_id == SUPP_103_ENTROPY_TRAP_15_TEMP_EXPR_ID) 
        {
            string env_id = D_CHAIN_10_ENV_ID;
            if (expr_id == SUPP_101_MOD_DCHAIN_10_TEMP_EXPR_ID) {
                env_id = MOD_D_CHAIN_10_ENV_ID;
            } else if (expr_id == SUPP_102_ENTROPY_TRAP_10_TEMP_EXPR_ID) {
                env_id = ENTROPY_TRAP_10_ENV_ID;
            } else if (expr_id == SUPP_103_ENTROPY_TRAP_15_TEMP_EXPR_ID) {
                env_id = ENTROPY_TRAP_15_ENV_ID;
            }
            int max_trial_length = ENV_ID_MAX_TRIAL_LEN.at(env_id);
            time_t expr_timestamp = std::time(nullptr);
            bool eval_wrt_time = false;
            double search_runtime = (expr_id == SUPP_103_ENTROPY_TRAP_15_TEMP_EXPR_ID) ? 100000 : 5000;
            double eval_delta = 50;
            int rollouts_per_mc_eval = 1; // det env
            int num_repeats = 25;
            int num_threads = 8;
            int eval_threads = 1; // det env

            // UCT run ids 
            vector<double> biases_to_try = {
                0.001,
                0.01,
                0.1,
                1.0,
                10.0,
                100.0,
            };

            for (double bias : biases_to_try) {
                unordered_map<string,double> alg_params =
                {
                    {BIAS_PARAM_ID, bias},
                };
                run_ids->push_back(RunID(
                    env_id,
                    expr_id,
                    expr_timestamp,
                    UCT_ALG_ID,
                    alg_params,
                    eval_wrt_time,
                    search_runtime,
                    eval_delta,
                    rollouts_per_mc_eval,
                    max_trial_length,
                    num_repeats,
                    num_threads,
                    eval_threads
                ));
            }
            
            // MENTS/DENTS/BTS run ids
            vector<double> temps_to_try = {
                0.001,
                0.0018,
                0.0032,
                0.0058,
                0.01,
                0.018,
                0.032,
                0.058,
                0.1,
                0.18,
                0.32,
                0.58,
                1.0,
                1.8,
                3.2,
                5.8,
                10.0,
                18.0,
                32.0,
                58.0,
                100.0,
            };
            vector<string> alg_ids = 
            {
                MENTS_ALG_ID,
                BTS_ALG_ID,
                DENTS_ALG_ID,
            };

            for (double temp : temps_to_try) {
                unordered_map<string,double> alg_params =
                {
                    {TEMP_PARAM_ID, temp},
                    {DECAY_FN_PARAM_ID, DECAY_FN_CONST},
                    {DECAY_FN_SCALE_PARAM_ID, 1.0},
                    {ENTROPY_COEFF_PARAM_ID, temp},
                    {ENTROPY_DECAY_FN_PARAM_ID, DECAY_FN_CONST},
                    {ENTROPY_DECAY_FN_SCALE_PARAM_ID, 1.0},
                    {EPSILON_PARAM_ID, 0.01},
                };
                for (string alg_id : alg_ids) {
                    run_ids->push_back(RunID(
                        env_id,
                        expr_id,
                        expr_timestamp,
                        alg_id,
                        alg_params,
                        eval_wrt_time,
                        search_runtime,
                        eval_delta,
                        rollouts_per_mc_eval,
                        max_trial_length,
                        num_repeats,
                        num_threads,
                        eval_threads
                    ));
                }
            }
            
            return run_ids;
        }

        // ----
        // expr_id: 110_uct_on_fl_dense / 111_uct_on_fl_sparse_len / 112_uct_on_fl_sparse_discounted 
        // sanity check UCT on frozen lake stuff??
        // ----
        if (expr_id == SUPP_110_UCT_ON_FL_DENSE
            || expr_id == SUPP_111_UCT_ON_FL_SPARSE_LEN
            || expr_id == SUPP_112_UCT_ON_FL_SPARSE_DISCOUNTED) 
        {
            double default_q_value = -50;
            string env_id = FROZEN_LAKE_NO_HOLE_DENSE_ENV_ID;
            if (expr_id == SUPP_111_UCT_ON_FL_SPARSE_LEN) {
                env_id = FROZEN_LAKE_NO_HOLE_SPARSE_LEN_ENV_ID;
            } else if (expr_id == SUPP_112_UCT_ON_FL_SPARSE_DISCOUNTED) {
                env_id = FROZEN_LAKE_NO_HOLE_SPARSE_DISCOUNTED_ENV_ID;
                default_q_value = 0;
            }
            int max_trial_length = ENV_ID_MAX_TRIAL_LEN.at(env_id);
            time_t expr_timestamp = std::time(nullptr);
            bool eval_wrt_time = false;
            double search_runtime = 5000;
            double eval_delta = 50;
            int rollouts_per_mc_eval = 1; // det env
            int num_repeats = 25;
            int num_threads = 8;
            int eval_threads = 1; // det env
            
            // UCT run ids 
            vector<double> biases_to_try = {
                // UctManagerArgs::bias_default,
                0.001,
                0.01,
                0.1,
                0.18,
                0.32,
                0.58,
                1.0,
                1.8,
                3.2,
                5.8,
                10.0,
                18.0,
                32.0,
                58.0,
                100.0,
                1000.0,
                10000.0,
            };

            for (double bias : biases_to_try) {
                unordered_map<string,double> alg_params =
                {
                    {BIAS_PARAM_ID, bias},
                };
                run_ids->push_back(RunID(
                    env_id,
                    expr_id,
                    expr_timestamp,
                    UCT_ALG_ID,
                    alg_params,
                    eval_wrt_time,
                    search_runtime,
                    eval_delta,
                    rollouts_per_mc_eval,
                    max_trial_length,
                    num_repeats,
                    num_threads,
                    eval_threads
                ));
            }
            
            // MENTS/DENTS/BTS run ids 
            vector<double> temps_to_try = {
                0.001,
                0.01,
                0.018,
                0.032,
                0.058,
                0.1,
                0.18,
                0.32,
                0.58,
                1.0,
                1.8,
                3.2,
                5.8,
                10.0,
                18.0,
                32.0,
                58.0,
                100.0,
                1000.0,
                10000.0,
            };
            vector<string> alg_ids = 
            {
                MENTS_ALG_ID,
                BTS_ALG_ID,
                DENTS_ALG_ID,
            };

            for (double temp : temps_to_try) {
                unordered_map<string,double> alg_params =
                {
                    {TEMP_PARAM_ID, temp},
                    {DECAY_FN_PARAM_ID, DECAY_FN_CONST},
                    {DECAY_FN_SCALE_PARAM_ID, 1.0},
                    {ENTROPY_COEFF_PARAM_ID, temp},
                    {ENTROPY_DECAY_FN_PARAM_ID, DECAY_FN_CONST},
                    {ENTROPY_DECAY_FN_SCALE_PARAM_ID, 1.0},
                    {EPSILON_PARAM_ID, 0.01},
                    {DEFAULT_Q_VALUE_PARAM_ID, default_q_value}
                };
                for (string alg_id : alg_ids) {
                    run_ids->push_back(RunID(
                        env_id,
                        expr_id,
                        expr_timestamp,
                        alg_id,
                        alg_params,
                        eval_wrt_time,
                        search_runtime,
                        eval_delta,
                        rollouts_per_mc_eval,
                        max_trial_length,
                        num_repeats,
                        num_threads,
                        eval_threads
                    ));
                }
            }
            
            return run_ids;
        }




        stringstream ss;
        ss << "Error in get_run_ids_from_expr_id for expr_id = " << expr_id;
        throw runtime_error(ss.str());
    }

    /**
     * Hyperparam optimiser - constructor
     */
    HyperparamOptimiser::HyperparamOptimiser(
        string env_id,
        string expr_id,
        time_t expr_timestamp,
        string alg_id,
        unordered_map<string, pair<double,double>> alg_params_min_max,
        bool eval_wrt_time,
        double search_runtime,
        int max_trial_length,
        double eval_delta,
        int rollouts_per_mc_eval,
        int num_repeats,
        int num_threads,
        int eval_threads,
        bayesopt::Parameters params,
        ofstream &results_fs) :
            bayesopt::ContinuousModel(RELEVANT_PARAM_IDS.at(alg_id).size(), params),
            num_hyperparams(RELEVANT_PARAM_IDS.at(alg_id).size()),
            env_id(env_id),
            expr_id(expr_id),
            expr_timestamp(expr_timestamp),
            alg_id(alg_id),
            alg_param_ids(RELEVANT_PARAM_IDS.at(alg_id)),
            alg_params_min_max(alg_params_min_max),
            eval_wrt_time(eval_wrt_time),
            search_runtime(search_runtime),
            max_trial_length(max_trial_length),
            eval_delta(eval_delta),
            rollouts_per_mc_eval(rollouts_per_mc_eval),
            num_repeats(num_repeats),
            num_threads(num_threads),
            eval_threads(eval_threads),
            num_envs((eval_threads > num_threads) ? eval_threads : num_threads),
            best_eval(numeric_limits<double>::lowest()),
            best_alg_params(),
            results_fs(results_fs),
            hp_opt_iter(0)
    {
        // error checking
        if (alg_param_ids.size() != alg_params_min_max.size()) {
            throw runtime_error("Expecting list of param min/max values to be same size as list of params for alg");
        }
        for (string param_id : alg_param_ids) {
            if (!alg_params_min_max.contains(param_id)) {
                stringstream ss;
                ss << "Expected list of hyperparams for alg_id=" << alg_id 
                    << " did not match keys provided in alg_params_min_max. Specifically the param_id=" << param_id 
                    << " was missing.";
                throw runtime_error(ss.str());
            }
        }

        // might as well set bounding box here
        bayesopt::vectord min_vec(num_hyperparams);
        bayesopt::vectord max_vec(num_hyperparams);
        for (size_t i=0; i<alg_param_ids.size(); i++) {
            pair<double,double> min_max = alg_params_min_max[alg_param_ids[i]];
            bool use_log_scale = (LOG_SCALE_PARAM_IDS.contains(alg_param_ids[i]));
            min_vec[i] = use_log_scale ? log(min_max.first) : min_max.first;
            max_vec[i] = use_log_scale ? log(min_max.second) : min_max.second;
        }
        bayesopt::ContinuousModel::setBoundingBox(min_vec,max_vec);
    };

    bool HyperparamOptimiser::is_python_env() 
    {
        return thts::is_python_env(env_id);
    }

    unordered_map<string, double> HyperparamOptimiser::get_alg_params_from_bayesopt_vec(bayesopt::vectord vec)
    {
        unordered_map<string, double> alg_params;
        for (size_t i=0; i<alg_param_ids.size(); i++) {
            string param_id = alg_param_ids[i];
            if (BOOLEAN_PARAM_IDS.contains(param_id)) {
                pair<double,double> min_max = alg_params_min_max[param_id]; 
                alg_params[param_id] = get_bool_val_from_cts_sample(vec[i], min_max.first, min_max.second);
            } else if (INTEGER_PARAM_IDS.contains(param_id)) {
                pair<double,double> min_max = alg_params_min_max[param_id]; 
                alg_params[param_id] = get_int_val_from_cts_sample(vec[i], min_max.first, min_max.second);
            } else {
                bool log_scaled = (LOG_SCALE_PARAM_IDS.contains(param_id));
                alg_params[param_id] = log_scaled ? exp(vec[i]) : vec[i];
            }
        }
        return alg_params;
    };

    bool HyperparamOptimiser::get_bool_val_from_cts_sample(double sample_val, int min, int max)
    {
        double midpoint = ((double) min+max) / 2.0;
        return (sample_val > midpoint);
    };

    int HyperparamOptimiser::get_int_val_from_cts_sample(double sample_val, int min, int max)
    {
        if (sample_val == max) {
            return max-1;            
        }
        return (int)sample_val;
    };

    /**
     * Hyperparam optimiser - fn to optimise
     */
    double HyperparamOptimiser::evaluateSample(const bayesopt::vectord &query) 
    {
        unordered_map<string,double> alg_params = get_alg_params_from_bayesopt_vec(query);
        RunID run_id(
            env_id,
            expr_id,
            expr_timestamp,
            alg_id,
            alg_params,
            eval_wrt_time,
            search_runtime,
            eval_delta,
            rollouts_per_mc_eval,
            max_trial_length,
            num_repeats,
            num_threads,
            eval_threads
        );
        double eval = thts::run_expr(run_id, false);
        if (eval > best_eval) {
            best_eval = eval;
            best_alg_params = alg_params;
        }
        write_eval_line(alg_params, eval);
        // bayes opt tried to minimise, so return *-1.0 because want to maximise
        return -1.0 * eval;
    };

    /**
     * Writes a header with the params for each eval top results_fs
     */
    void HyperparamOptimiser::write_header()
    {
        // expr params
        results_fs 
            << "env_id,alg_id,search_runtime,max_trial_length,rollouts_per_mc_eval,num_repeats,num_threads" << endl;
        
        results_fs 
            << env_id << ","
            << alg_id << ","
            << search_runtime << ","
            << max_trial_length << ","
            << rollouts_per_mc_eval << ","
            << num_repeats << ","
            << num_threads 
            << endl << endl;
        
        // hyperparams (with sample number (hp_opt_iter) and eval at start/end)
        results_fs << "hp_opt_iter,";
        for (string& param_id : alg_param_ids) {
            results_fs << param_id << ",";
        } 
        results_fs << "eval(mc_estimate_expected_utility),best_eval_so_far" << endl;

        // Print out the min an max params trying
        results_fs << "MIN,";
        for (string param_id : alg_param_ids) {
            results_fs << alg_params_min_max[param_id].first << ",";
        } 
        results_fs << "MIN" << endl;
        results_fs << "MAX,";
        for (string param_id : alg_param_ids) {
            results_fs << alg_params_min_max[param_id].second << ",";
        } 
        results_fs << "MAX" << endl;

    };

    /**
     * Write eval/hyperparam sample line to file
     * - note that hp_opt_iter only used here, and also updated here
     */
    void HyperparamOptimiser::write_eval_line(unordered_map<string,double> alg_params, double eval)
    {   
        results_fs << hp_opt_iter++ << ",";
        for (string param_id : alg_param_ids) {
            results_fs << alg_params[param_id] << ",";
        }
        results_fs << eval << "," << best_eval << endl;
    };

    void HyperparamOptimiser::write_best_eval()
    {
        results_fs << endl;
        results_fs << "Best eval with params:" << endl;
        results_fs << "eval (mc_estimate_expected_utility) = " << best_eval << endl;
        for (pair<string,double> pr : best_alg_params) {
            results_fs << pr.first << " = " << pr.second << endl;
        }
    };

    /**
     * Gets hyperparam optimiser from expr_id
     */
    shared_ptr<HyperparamOptimiser> get_hyperparam_optimiser_from_expr_id(
        string expr_id, time_t expr_timestamp, ofstream &hp_opt_fs)
    {
        // Params shared across optimisations (related to envs)
        string env_id = HP_OPT_EXPR_ID_TO_ENV_ID.at(expr_id);
        bool eval_wrt_time = false;
        double search_runtime = 50000.0;
        int max_trial_length = ENV_ID_MAX_TRIAL_LEN.at(env_id);
        double eval_delta = 25000.0;
        int rollouts_per_mc_eval = (DET_ENVS.contains(env_id)) ? 1 : 1024;
        int num_repeats = 5;
        int num_threads = 16;
        int eval_threads = 16;

        // Params being tuned
        string alg_id;
        unordered_map<string, pair<double,double>> alg_params_min_max;

        // UCT
        if (expr_id == HP_OPT_600_UCT_EXPR_ID 
            || expr_id == HP_OPT_601_UCT_EXPR_ID
            || expr_id == HP_OPT_602_UCT_EXPR_ID
            || expr_id == HP_OPT_603_UCT_EXPR_ID) 
        {
            alg_id = UCT_ALG_ID;
            alg_params_min_max = {
                {ADAPTIVE_BIAS_PARAM_ID, make_pair(0.0, 1.0)},
                {BIAS_PARAM_ID, make_pair(0.001, 1000.0)},
            };
        }
        // MaxUCT
        else if (expr_id == HP_OPT_610_MAX_UCT_EXPR_ID
            || expr_id == HP_OPT_611_MAX_UCT_EXPR_ID
            || expr_id == HP_OPT_612_MAX_UCT_EXPR_ID
            || expr_id == HP_OPT_613_MAX_UCT_EXPR_ID)
        {
            alg_id = MAX_UCT_ALG_ID;
            alg_params_min_max = {
                {ADAPTIVE_BIAS_PARAM_ID, make_pair(0.0, 1.0)},
                {BIAS_PARAM_ID, make_pair(0.001, 1000.0)},
            };
        }
        // MENTS
        else if (expr_id == HP_OPT_620_MENTS_EXPR_ID
            || expr_id == HP_OPT_621_MENTS_EXPR_ID
            || expr_id == HP_OPT_622_MENTS_EXPR_ID
            || expr_id == HP_OPT_623_MENTS_EXPR_ID)
        {   
            double default_q_value = -double(max_trial_length);
            alg_id = MENTS_ALG_ID;
            alg_params_min_max = {
                {NORMALISE_Q_VALUES_PARAM_ID, make_pair(0.0, 1.0)},
                {TEMP_PARAM_ID, make_pair(0.001, 1000.0)},
                {EPSILON_PARAM_ID, make_pair(0.0, 10.0)},
                {DEFAULT_Q_VALUE_PARAM_ID, make_pair(default_q_value,default_q_value)}
            };
        }
        // BTS
        else if (expr_id == HP_OPT_630_BTS_EXPR_ID
            || expr_id == HP_OPT_631_BTS_EXPR_ID
            || expr_id == HP_OPT_632_BTS_EXPR_ID
            || expr_id == HP_OPT_633_BTS_EXPR_ID)
        {
            double default_q_value = -double(max_trial_length);
            alg_id = BTS_ALG_ID;
            alg_params_min_max = {
                {NORMALISE_Q_VALUES_PARAM_ID, make_pair(0.0, 1.0)},
                {TEMP_PARAM_ID, make_pair(0.001, 1000.0)},
                {DECAY_FN_PARAM_ID, make_pair(0.0, 3.0)},
                {DECAY_FN_SCALE_PARAM_ID, make_pair(0.01, 100.0)},
                {EPSILON_PARAM_ID, make_pair(0.0, 10.0)},
                {DEFAULT_Q_VALUE_PARAM_ID, make_pair(default_q_value,default_q_value)}
            };
        }
        // DENTS
        else if (expr_id == HP_OPT_640_DENTS_EXPR_ID
            || expr_id == HP_OPT_641_DENTS_EXPR_ID
            || expr_id == HP_OPT_642_DENTS_EXPR_ID
            || expr_id == HP_OPT_643_DENTS_EXPR_ID)
        {
            double default_q_value = -double(max_trial_length);
            alg_id = DENTS_ALG_ID;
            alg_params_min_max = {
                {NORMALISE_Q_VALUES_PARAM_ID, make_pair(0.0, 1.0)},
                {TEMP_PARAM_ID, make_pair(0.001, 1000.0)},
                {DECAY_FN_PARAM_ID, make_pair(0.0, 3.0)},
                {DECAY_FN_SCALE_PARAM_ID, make_pair(0.01, 100.0)},
                {ENTROPY_COEFF_PARAM_ID, make_pair(0.001, 1000.0)},
                {ENTROPY_DECAY_FN_PARAM_ID, make_pair(0.0, 3.0)},
                {ENTROPY_DECAY_FN_SCALE_PARAM_ID, make_pair(0.01, 100.0)},
                {EPSILON_PARAM_ID, make_pair(0.0, 10.0)},
                {DEFAULT_Q_VALUE_PARAM_ID, make_pair(default_q_value,default_q_value)}
            };
        }
        // RENTS
        else if (expr_id == HP_OPT_650_RENTS_EXPR_ID
            || expr_id == HP_OPT_651_RENTS_EXPR_ID
            || expr_id == HP_OPT_652_RENTS_EXPR_ID
            || expr_id == HP_OPT_653_RENTS_EXPR_ID)
        {
            double default_q_value = -double(max_trial_length);
            alg_id = RENTS_ALG_ID;
            alg_params_min_max = {
                {NORMALISE_Q_VALUES_PARAM_ID, make_pair(0.0, 1.0)},
                {TEMP_PARAM_ID, make_pair(0.001, 1000.0)},
                {EPSILON_PARAM_ID, make_pair(0.0, 10.0)},
                {DEFAULT_Q_VALUE_PARAM_ID, make_pair(default_q_value,default_q_value)}
            };
        }
        // TENTS
        else if (expr_id == HP_OPT_660_TENTS_EXPR_ID
            || expr_id == HP_OPT_661_TENTS_EXPR_ID
            || expr_id == HP_OPT_662_TENTS_EXPR_ID
            || expr_id == HP_OPT_663_TENTS_EXPR_ID)
        {
            double default_q_value = -double(max_trial_length);
            alg_id = TENTS_ALG_ID;
            alg_params_min_max = {
                {NORMALISE_Q_VALUES_PARAM_ID, make_pair(0.0, 1.0)},
                {TEMP_PARAM_ID, make_pair(0.001, 1000.0)},
                {EPSILON_PARAM_ID, make_pair(0.0, 10.0)},
                {DEFAULT_Q_VALUE_PARAM_ID, make_pair(default_q_value,default_q_value)}
            };
        }
        // HMCTS
        else if (expr_id == HP_OPT_670_HMCTS_EXPR_ID
            || expr_id == HP_OPT_671_HMCTS_EXPR_ID
            || expr_id == HP_OPT_672_HMCTS_EXPR_ID
            || expr_id == HP_OPT_673_HMCTS_EXPR_ID)
        {
            alg_id = HMCTS_ALG_ID;
            alg_params_min_max = {
                {ADAPTIVE_BIAS_PARAM_ID, make_pair(0.0, 1.0)},
                {UCT_BUDGET_PARAM_ID, make_pair(1.0, 5000.0)},
            };
        }
        // Default, haven't set up hp opt experiments for this env
        else 
        {
            stringstream ss;
            ss << "Error in get_hyperparam_optimiser_from_expr_id for expr_id = " << expr_id;
            throw runtime_error(ss.str());
        }

        // Bayesopt params
        bayesopt::Parameters bo_params;
        bo_params.surr_name = "sGaussianProcessML";
        bo_params.noise = 1.0; 
        bo_params.n_iterations = 190;
        bo_params.n_init_samples = 10;
        bo_params.n_iter_relearn = 10;
        bo_params.verbose_level = 0;

        return make_shared<HyperparamOptimiser>(
            env_id,
            expr_id,
            expr_timestamp,
            alg_id,
            alg_params_min_max,
            eval_wrt_time,
            search_runtime,
            max_trial_length,
            eval_delta,
            rollouts_per_mc_eval,
            num_repeats,
            num_threads,
            eval_threads,
            bo_params,
            hp_opt_fs
        );
    };

    /**
     * Lookup expr_id from prefix
     */
    string lookup_expr_id_from_prefix(string expr_id_prefix) 
    {
        for (const string& expr_id : ALL_EXPR_IDS) {
            if (expr_id.starts_with(expr_id_prefix)) {
                return expr_id;
            }
        }
        throw runtime_error(
            "Error looking up expr_id from prefix. Either forgot to add expr_id to 'ALL_EXPR_IDS' list or typo?");
    }
    
    /**
     * Helper to make a string of:
     * "param1=val1/param2=val2/.../paramN=valN/"
     * Old version output:
     * "param1=val1,param2=val2,...,paramN=valN",
     * but lead to filenames that were too long
    */
    string get_params_string_helper(RunID& run_id) {
        stringstream ss;
        const vector<string> &relevant_param_ids = RELEVANT_PARAM_IDS.at(run_id.alg_id);
        unordered_set<string> relevant_param_ids_set(relevant_param_ids.begin(),relevant_param_ids.end());
        for (pair<string,double> param_val_entry : run_id.alg_params) {
            if (!relevant_param_ids_set.contains(param_val_entry.first)) {
                continue;
            }
            ss << param_val_entry.first << "=" << param_val_entry.second << "/";
        }
        return ss.str();
    }

    /**
     * Gets the results directory for this run (doesn't check/make)
    */
    string get_results_dir(RunID& run_id) {
        stringstream ss;
        ss << "results/" 
            << run_id.expr_id << "_" << run_id.expr_timestamp << "/" 
            << run_id.env_id << "/" 
            << run_id.alg_id << "/"
            << get_params_string_helper(run_id);
        return ss.str();
    }
    
    /**
     * Checks if env corresponding to 'env_id' is a python env
     */
    bool is_python_env(string env_id) 
    {
        return (PY_ENVS.contains(env_id) 
            || GYM_ENVS.contains(env_id));
    }

    /**
     * Create and return the env
    */
    shared_ptr<ThtsEnv> get_env(RunID& run_id) 
    {
        string thts_unique_filename = get_results_dir(run_id);
        string& env_id = run_id.env_id;

        if (GYM_ENVS.contains(env_id)) {
            shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
            return make_shared<GymMultiprocessingThtsEnv>(pickle_wrapper, thts_unique_filename, env_id);
        }

        if (env_id == D_CHAIN_10_ENV_ID)
        {
            return make_shared<DChainEnv>(10,1.0);
        }
        if (env_id == MOD_D_CHAIN_10_ENV_ID)
        {
            return make_shared<DChainEnv>(10,0.5); 
        }

        if (env_id == ENTROPY_TRAP_10_ENV_ID)
        {
            return make_shared<EntropyTrapEnv>(10,10,1.0);
        }

        if (env_id == ENTROPY_TRAP_15_ENV_ID)
        {
            return make_shared<EntropyTrapEnv>(15,15,1.0);
        }

        if (env_id == FROZEN_LAKE_NO_HOLE_DENSE_ENV_ID || env_id == FROZEN_LAKE_NO_HOLE_SPARSE_LEN_ENV_ID || env_id == FROZEN_LAKE_NO_HOLE_SPARSE_DISCOUNTED_ENV_ID)
        {
            int reward_type = FL_DENSE_REWARD;
            if (env_id == FROZEN_LAKE_NO_HOLE_SPARSE_LEN_ENV_ID) {
                reward_type = FL_SPARSE_LEN_REWARD;
            } else if (env_id == FROZEN_LAKE_NO_HOLE_SPARSE_DISCOUNTED_ENV_ID) {
                reward_type = FL_SPARSE_DISCOUNTED_REWARD;
            }
            return make_shared<FrozenLakeEnv>(6,6,FL_6x6_NO_HOLE_MAP,false,reward_type);
        }

        if (env_id == FROZEN_LAKE_D_8x8_ENV_ID)
        {
            return make_shared<FrozenLakeEnv>(8,8,FL_8x8_MAP,false,FL_DENSE_REWARD);
        }
        if (env_id == FROZEN_LAKE_S_8x8_ENV_ID)
        {
            return make_shared<FrozenLakeEnv>(8,8,FL_8x8_MAP,false,FL_SPARSE_DISCOUNTED_REWARD);
        }

        if (env_id == FROZEN_LAKE_D_8x16_ENV_ID)
        {
            return make_shared<FrozenLakeEnv>(8,16,FL_GEN_8x16_MAP,false,FL_DENSE_REWARD);
        }
        if (env_id == FROZEN_LAKE_S_8x16_ENV_ID)
        {
            return make_shared<FrozenLakeEnv>(8,16,FL_GEN_8x16_MAP,false,FL_SPARSE_DISCOUNTED_REWARD);
        }

        if (env_id == FROZEN_LAKE_D_16x16_ENV_ID)
        {
            return make_shared<FrozenLakeEnv>(8,16,FL_GEN_16x16_MAP,false,FL_DENSE_REWARD);
        }
        if (env_id == FROZEN_LAKE_S_16x16_ENV_ID)
        {
            return make_shared<FrozenLakeEnv>(8,16,FL_GEN_16x16_MAP,false,FL_SPARSE_DISCOUNTED_REWARD);
        }

        if (env_id == SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID)
        {
            return make_shared<FrozenLakeEnv>(4,4,FL_4x4_MAP,true,FL_DENSE_REWARD);
        }
        if (env_id == SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID)
        {
            return make_shared<FrozenLakeEnv>(4,4,FL_4x4_MAP,true,FL_SPARSE_DISCOUNTED_REWARD,1.0);
        }

        if (env_id == SLIPPY_FROZEN_LAKE_D_5x5_ENV_ID)
        {
            return make_shared<FrozenLakeEnv>(5,5,FL_GEN_5x5_MAP,true,FL_DENSE_REWARD);
        }
        if (env_id == SLIPPY_FROZEN_LAKE_S_5x5_ENV_ID)
        {
            return make_shared<FrozenLakeEnv>(5,5,FL_GEN_5x5_MAP,true,FL_SPARSE_DISCOUNTED_REWARD,1.0);
        }

        if (env_id == SLIPPY_FROZEN_LAKE_D_6x6_ENV_ID)
        {
            return make_shared<FrozenLakeEnv>(5,5,FL_GEN_6x6_MAP,true,FL_DENSE_REWARD);
        }
        if (env_id == SLIPPY_FROZEN_LAKE_S_6x6_ENV_ID)
        {
            return make_shared<FrozenLakeEnv>(5,5,FL_GEN_6x6_MAP,true,FL_SPARSE_DISCOUNTED_REWARD,1.0);
        }

        if (env_id == SAILING_ENV_NORTH_ID)
        {
            return make_shared<SailingEnv>(8,8,NN);
        }
        
        if (env_id == SAILING_ENV_SOUTH_EAST_ID)
        {
            return make_shared<SailingEnv>(8,8,SE);
        }

        stringstream ss;
        ss << "Error in get_env for env_id = " << env_id;
        throw runtime_error(ss.str());
    }

}