#include "main_aux/run_manager.h"

// #include "main_aux/run_expr.h"

// #include "helper.h"

#include "algorithms/uct/uct_manager.h"
#include "algorithms/uct/hmcts_manager.h"
#include "algorithms/ments/ments_manager.h"
#include "algorithms/ments/dents/dents_manager.h"

#include "algorithms/uct/uct_decision_node.h"
#include "algorithms/uct/max_uct_decision_node.h"
#include "algorithms/ments/ments_decision_node.h"
#include "algorithms/est/est_decision_node.h"
#include "algorithms/ments/dents/dents_decision_node.h"
#include "algorithms/uct/hmcts_decision_node.h"
#include "algorithms/uct/max_uct_decision_node.h"
#include "algorithms/ments/rents/rents_decision_node.h"
#include "algorithms/ments/tents/tents_decision_node.h"

#include "algorithms/common/decaying_temp.h"

#include "py/pickle_wrapper.h"
// #include "py/py_multiprocessing_thts_env.h"
// #include "py/gym_multiprocessing_thts_env.h"

#include "main_aux/envs/d_chain.h"
#include "main_aux/envs/entropy_trap.h"
#include "main_aux/envs/frozen_lake.h"
#include "main_aux/envs/sailing.h"

// #include <cmath>

#include <vector>
#include <stdexcept>
#include <sstream>
#include <string>

using namespace std;
using namespace thts;
// using namespace thts::python;

// namespace py = pybind11;

namespace thts {

    /**
     * Constructor
     */
    RunManager::RunManager(std::time_t xpr_timestamp, ConfigMap xpr_config, ConfigMap alg_config) :
        xpr_timestamp(xpr_timestamp), xpr_config(xpr_config), alg_config(alg_config)
    {
        validate_config_or_raise_exception();
    }

    /**
     *  Checks all expected params are present, and that no additional params
     */  
    RunManager::validate_config_or_raise_exception()
    {
        if (xpr_config.size() != 12)
        {
            throw runtime_error("Expecting 12 entries in the xpr level config.");
        }

        vector<string> xpr_param_ids = 
        {
            XPR_PARAM_ID_NAME, 
            XPR_PARAM_ID_ENV, 
            XPR_PARAM_ID_MCTS_MODE, 
            XPR_PARAM_ID_MAX_TRIAL_LENGTH,
            XPR_PARAM_ID_RUNTIME_BOUNDED, 
            XPR_PARAM_ID_TERMINATION_BOUND, 
            XPR_PARAM_ID_REPEATED_RUNS_PER_ALG, 
            XPR_PARAM_ID_SEARCH_THREADS, 
            XPR_PARAM_ID_EVAL_DELTA, 
            XPR_PARAM_ID_EVAL_ROLLOUTS, 
            XPR_PARAM_ID_EVAL_THREADS,
        };

        for (string& xpr_param_id : xpr_param_ids) 
        {
            if (!xpr_config.contains(xpr_param_id))
            {
                stringstream ss;
                ss << "Expecting to find value for " << xpr_param_id << " in xpr level config.";
                throw runtime_error(ss.str());
            }
        }

        string alg_id = get_config_value(alg_config, XPR_OR_ALG_ID_TAG);

        vector<string> alg_ids =
        {
            ALG_ID_UCT, 
            ALG_ID_MAX_UCT, 
            ALG_ID_HMCTS, 
            ALG_ID_MENTS, 
            ALG_ID_RENTS, 
            ALG_ID_TENTS, 
            ALG_ID_BTS, 
            ALG_ID_DENTS,
        };

        if (!alg_ids.contains(alg_id))
        {
            stringstream ss;
            ss << "Found unrecognised algorithm id " << alg_id << " in alg level config.";
            throw runtime_error(ss.str());
        }

        vector<string> param_ids_expecting = ALG_ID_TO_ALG_PARAM_IDS[alg_id];
        for (string& param_id : param_ids_expecting) 
        {
            if (!alg_config.contains(param_id)) 
            {
                stringstream ss;
                ss << "Expecting to find value for " << param_id << " in alg level config for " << alg_id << ".";
                throw runtime_error(ss.str());

            }
        }
    }

    /**
     * Lookup config from xpr_id_prefix, so unique id's, but not pain to type
     */
    vector<ConfigMap> RunManager::lookup_config_vector_from_xpr_prefix(string xpr_id_prefix)
    {
        // Validate that all configs have the first ConfigMap with xpr level config, including an xpr_id
        for (vector<ConfigMap>& config : ALL_CONFIGS)
        {
            ConfigMap& xpr_config = config[0];
            if (get_config_value(xpr_config, XPR_OR_ALG_ID_TAG) != XPR_PARAMS_ID_TAG)
            {
                throw runtime_error("Expecting first map in each config (vector) to specify xpr level config with correct tagging.");
            }
            if (!xpr_config.contains(XPR_PARAM_ID_NAME)) 
            {
                throw runtime_error("Expecting xpr level config to specify an xpr_name");
            }
        }

        // Lookup
        for (vector<ConfigMap>& config : ALL_CONFIGS)
        {
            string& xpr_name = get_config_value(config[0], XPR_PARAM_ID_NAME);
            if (xpr_name.starts_with(xpr_id_prefix))
            {
                return config;
            }
        }

        stringstream ss;
        ss << "Error looking up xpr_id from prefix, couldn't find config starting with " << xpr_id_prefix << "in ALL_CONFIGS";
        throw runtime_error(ss.str());
    }

    /**
     * Config -> RunManagers
     */
    shared_ptr<vector<RunManager>> RunManager::get_run_managers_from_config_vector(vector<ConfigMap>& config_vector)
    {
        time_t xpr_timestamp = std::time(nullptr);
        shared_ptr<vector<RunManager>> run_managers = std::make_shared<vector<RunManager>>();
        for (size_t i=1; i<config_vector.size(); i++)
        {  
            run_managers->push_back(RunManager(xpr_timestamp, config_vector[0], config_vector[i]));
        }
        return run_managers;
    }


    /**
     * Getters - xpr level config
     */
    string RunManager::get_xpr_name()           { return get_config_value(xpr_config, XPR_PARAM_ID_NAME); }
    string RunManager::get_env_id()             { return get_config_value(xpr_config, XPR_PARAM_ID_ENV); }
    bool RunManager::get_mcts_mode()            { return get_config_value(xpr_config, XPR_PARAM_ID_MCTS_MODE); }
    int RunManager::get_max_trial_length()      { return get_config_value(xpr_config, XPR_PARAM_ID_MAX_TRIAL_LENGTH); }
    bool RunManager::xpr_is_runtime_bounded()   { return get_config_value(xpr_config, XPR_PARAM_ID_RUNTIME_BOUNDED); }
    double RunManager::get_termination_bound()  { return get_config_value(xpr_config, XPR_PARAM_ID_TERMINATION_BOUND); }
    int RunManager::get_repeated_runs_per_alg() { return get_config_value(xpr_config, XPR_PARAM_ID_REPEATED_RUNS_PER_ALG); }
    int RunManager::get_num_search_threads()    { return get_config_value(xpr_config, XPR_PARAM_ID_SEARCH_THREADS); }
    double RunManager::get_eval_delta()         { return get_config_value(xpr_config, XPR_PARAM_ID_EVAL_DELTA); }
    int RunManager::get_num_eval_rollouts()     { return get_config_value(xpr_config, XPR_PARAM_ID_EVAL_ROLLOUTS); }
    int RunManager::get_num_eval_threads()      { return get_config_value(xpr_config, XPR_PARAM_ID_EVAL_THREADS); }

    /**
     * Getters - alg level config
     */
    string RunManager::get_alg_id()             { return get_config_value(alg_config, XPR_OR_ALG_ID_TAG); }
    double RunManager::get_bias()               { return get_config_value(alg_config, ALG_PARAM_ID_BIAS); }
    int RunManager::get_uct_budget()            { return get_config_value(alg_config, ALG_PARAM_ID_UCT_BUDGET); }
    double RunManager::get_init_temp()          { return get_config_value(alg_config, ALG_PARAM_ID_INIT_TEMP); }
    double RunManager::get_temp_decay_rate()    { return get_config_value(alg_config, ALG_PARAM_ID_TEMP_DECAY_RATE); }
    double RunManager::get_init_entropy_coeff() { return get_config_value(alg_config, ALG_PARAM_ID_INIT_ENTROPY_COEFF); }
    double RunManager::get_entropy_zero_at()    { return get_config_value(alg_config, ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT); }
    double RunManager::get_epsilon()            { return get_config_value(alg_config, ALG_PARAM_ID_EPSILON); }
    double RunManager::get_default_q_value()    { return get_config_value(alg_config, ALG_PARAM_ID_DEFAULT_Q_VALUE); }


    /**
     * Helper to make a string of:
     * "param1=val1/param2=val2/.../paramN=valN/"
     * Old version output:
     * "param1=val1,param2=val2,...,paramN=valN",
     * but lead to filenames that were too long
    */
    string get_params_string_helper(RunManager& run_manager) {
        stringstream ss;
        const vector<string>& relevant_alg_param_ids = ALG_ID_TO_ALG_PARAM_IDS.at(get_alg_id());
        for (const string& alg_param_id : relevant_alg_param_ids)
        {
            ss << alg_param_id << "=" << get_config_value<double>(run_manager.alg_config, alg_param_id) << "/";
        }
        return ss.str();
    }

    /**
     * Gets the results directory for this run (doesn't check/make)
    */
    string RunManager::get_eval_logs_dir() {
        stringstream ss;
        ss << "eval_logs_aux/" 
            << get_xpr_name() << "_" << xpr_timestamp << "/"
            << get_env_id() << "/"
            << get_alg_id() << "/"
            << get_params_string_helper(*this);
        return ss.str();
    }


    /**
     * Returns if the env we are using is a python env
    */
    bool RunManager::is_python_env()
    {
        return (PY_ENVS.contains(get_env_id()) || GYM_ENVS.contains(get_env_id()));

    }

    /**
     * Returns an instance of ThtsEnv to use for this run
    */
    shared_ptr<ThtsEnv> RunManager::get_env()
    {
        string thts_unique_filename = get_results_dir(run_id);
        string& env_id = run_id.env_id;

        if (GYM_ENVS.contains(env_id)) {
            shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
            return make_shared<GymMultiprocessingThtsEnv>(pickle_wrapper, thts_unique_filename, env_id);
        }
        
        if (env_id == ENV_ID_D_CHAIN_10)        return make_shared<DCHainEnv>(10,1.0);
        if (env_id == ENV_ID_MOD_D_CHAIN_10)    return make_shared<DCHainEnv>(10,0.5);
        if (env_id == ENV_ID_ENTROPY_TRAP_10)   return make_shared<EntropyTrapEnv>(10,10,1.0);
        if (env_id == ENV_ID_ENTROPY_TRAP_15)   return make_shared<EntropyTrapEnv>(15,15,1.0);

        if (env_id == ENV_ID_FROZEN_LAKE_NO_HOLE_DENSE)             return make_shared<FrozenLakeEnv>(6,6,FL_6x6_NO_HOLE_MAP,false,FL_DENSE_REWARD);
        if (env_id == ENV_ID_FROZEN_LAKE_NO_HOLE_SPARSE_LEN)        return make_shared<FrozenLakeEnv>(6,6,FL_6x6_NO_HOLE_MAP,false,FL_SPARSE_LEN_REWARD);
        if (env_id == ENV_ID_FROZEN_LAKE_NO_HOLE_SPARSE_DISCOUNTED) return make_shared<FrozenLakeEnv>(6,6,FL_6x6_NO_HOLE_MAP,false,FL_SPARSE_DISCOUNTED_REWARD);

        if (env_id == ENV_ID_FROZEN_LAKE_S_8x8)     return make_shared<FrozenLakeEnv>(8,8,FL_8x8_MAP,false,FL_SPARSE_DISCOUNTED_REWARD);
        if (env_id == ENV_ID_FROZEN_LAKE_D_8x8)     return make_shared<FrozenLakeEnv>(8,8,FL_8x8_MAP,false,FL_DENSE_REWARD);
        if (env_id == ENV_ID_FROZEN_LAKE_D_8x16)    return make_shared<FrozenLakeEnv>(8,16,FL_GEN_8x16_MAP,false,FL_DENSE_REWARD);
        if (env_id == ENV_ID_FROZEN_LAKE_S_8x16)    return make_shared<FrozenLakeEnv>(8,16,FL_GEN_8x16_MAP,false,FL_SPARSE_DISCOUNTED_REWARD);
        if (env_id == ENV_ID_FROZEN_LAKE_D_16x16)   return make_shared<FrozenLakeEnv>(16,16,FL_GEN_16x16_MAP,false,FL_DENSE_REWARD);
        if (env_id == ENV_ID_FROZEN_LAKE_S_16x16)   return make_shared<FrozenLakeEnv>(16,16,FL_GEN_16x16_MAP,false,FL_SPARSE_DISCOUNTED_REWARD);

        if (env_id == ENV_ID_SLIPPY_FROZEN_LAKE_D_4x4) return make_shared<FrozenLakeEnv>(4,4,FL_4x4_MAP,true,FL_DENSE_REWARD);
        if (env_id == ENV_ID_SLIPPY_FROZEN_LAKE_S_4x4) return make_shared<FrozenLakeEnv>(4,4,FL_4x4_MAP,true,FL_SPARSE_DISCOUNTED_REWARD,1.0);
        if (env_id == ENV_ID_SLIPPY_FROZEN_LAKE_D_5x5) return make_shared<FrozenLakeEnv>(5,5,FL_GEN_5x5_MAP,true,FL_DENSE_REWARD);
        if (env_id == ENV_ID_SLIPPY_FROZEN_LAKE_S_5x5) return make_shared<FrozenLakeEnv>(5,5,FL_GEN_5x5_MAP,true,FL_SPARSE_DISCOUNTED_REWARD,1.0);
        if (env_id == ENV_ID_SLIPPY_FROZEN_LAKE_D_6x6) return make_shared<FrozenLakeEnv>(6,6,FL_GEN_6x6_MAP,true,FL_DENSE_REWARD);
        if (env_id == ENV_ID_SLIPPY_FROZEN_LAKE_S_6x6) return make_shared<FrozenLakeEnv>(6,6,FL_GEN_6x6_MAP,true,FL_SPARSE_DISCOUNTED_REWARD,1.0);

        if (env_id == ENV_ID_SAILING_NORTH_ID)             return make_shared<SailingEnv>(8,8,NN);
        if (env_id == ENV_ID_SAILING_SOUTH_EAST_ID)        return make_shared<SailingEnv>(8,8,SE);
        if (env_id == ENV_ID_SAILING_8x16_NORTH_ID)        return make_shared<SailingEnv>(8,16,NN);
        if (env_id == ENV_ID_SAILING_8x16_SOUTH_EAST_ID)   return make_shared<SailingEnv>(8,16,SE);
        if (env_id == ENV_ID_SAILING_16x16_NORTH_ID)       return make_shared<SailingEnv>(16,16,NN);
        if (env_id == ENV_ID_SAILING_16x16_SOUTH_EAST_ID)  return make_shared<SailingEnv>(16,16,SE);

        stringstream ss;
        ss << "Error in get_env for env_id = " << env_id;
        throw runtime_error(ss.str());
    }

    /**
     * Returns and instance of ThtsManager to use for this run
    */
    shared_ptr<ThtsManager> RunManager::get_thts_manager(shared_ptr<ThtsEnv> env)
    {
        
    }

    /**
     * Returns a root node to use for search given these params
    */
    shared_ptr<ThtsDNode> get_root_search_node(shared_ptr<ThtsEnv> env, shared_ptr<ThtsManager> manager)
    {
        if (alg_id == ALG_ID_UCT) {
            shared_ptr<UctManager> uct_manager = static_pointer_cast<UctManager>(manager);
            return make_shared<UctDNode>(uct_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_MAX_UCT) {
            shared_ptr<UctManager> uct_manager = static_pointer_cast<UctManager>(manager);
            return make_shared<MaxUctDNode>(uct_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_MENTS) {
            shared_ptr<MentsManager> ments_manager = static_pointer_cast<DentsManager>(manager);
            return make_shared<MentsDNode>(ments_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_BTS) {
            shared_ptr<DentsManager> bts_manager = static_pointer_cast<DentsManager>(manager);
            return make_shared<EstDNode>(bts_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_DENTS) {
            shared_ptr<DentsManager> dents_manager = static_pointer_cast<DentsManager>(manager);
            return make_shared<DentsDNode>(dents_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_RENTS) {
            shared_ptr<MentsManager> ments_manager = static_pointer_cast<MentsManager>(manager);
            return make_shared<RentsDNode>(ments_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_TENTS) {
            shared_ptr<MentsManager> ments_manager = static_pointer_cast<MentsManager>(manager);
            return make_shared<TentsDNode>(ments_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_HMCTS) {
            shared_ptr<HmctsManager> uct_manager = static_pointer_cast<HmctsManager>(manager);
            return make_shared<HmctsDNode>(uct_manager, env->get_initial_state_itfc(), 0, 0);
        }

        stringstream ss;
        ss << "Error in RunID get_root_search_node for alg_id = " << alg_id;
        throw runtime_error(ss.str());
    }

}


//     /**
//      * Initialised constructor
//     */
//     RunID::RunID(
//         string env_id,
//         string expr_id,
//         time_t expr_timestamp,
//         string alg_id,
//         unordered_map<string, double>& alg_params,
//         bool eval_wrt_time,
//         double search_runtime,
//         double eval_delta,
//         int rollouts_per_mc_eval,
//         int max_trial_length,
//         int num_repeats,
//         int num_threads,
//         int eval_threads,
//         bool mcts_mode) :
//             env_id(env_id),
//             expr_id(expr_id),
//             expr_timestamp(expr_timestamp),
//             alg_id(alg_id),
//             alg_params(alg_params),
//             mcts_mode(mcts_mode),
//             adaptive_bias(UctManagerArgs::adaptive_bias_default),
//             bias(UctManagerArgs::bias_default),
//             hmcts_uct_budget(HmctsManagerArgs::uct_budget_threshold_default),
//             normalise_q_values(MentsManagerArgs::normalise_q_values_default),
//             default_q_value(0.0),
//             epsilon(MentsManagerArgs::epsilon_default),
//             temp(MentsManagerArgs::temp_default),
//             decay_fn(DECAY_FN_CONST),
//             decay_fn_scale(1.0),
//             entropy_coeff(1.0),
//             entropy_decay_fn(DECAY_FN_CONST),
//             entropy_decay_fn_scale(1.0),
//             eval_wrt_time(eval_wrt_time),
//             search_runtime(search_runtime),
//             eval_delta(eval_delta),
//             rollouts_per_mc_eval(rollouts_per_mc_eval),
//             max_trial_length(max_trial_length),
//             num_repeats(num_repeats),
//             num_threads(num_threads),
//             eval_threads(eval_threads),
//             num_envs((eval_threads > num_threads) ? eval_threads : num_threads)
//     {
//         for (pair<string,double> pair : alg_params) {
//             string param_id = pair.first;
//             double param_val = pair.second;

//             if (param_id == MCTS_MODE_PARAM_ID) {
//                 mcts_mode = (bool) param_val;
//             } else if (param_id == ADAPTIVE_BIAS_PARAM_ID) {
//                 adaptive_bias = (bool) param_val;
//             } else if (param_id == BIAS_PARAM_ID) {
//                 bias = param_val;
//             } else if (param_id == UCT_BUDGET_PARAM_ID) {
//                 hmcts_uct_budget = (int) param_val;
//             } else if (param_id == NORMALISE_Q_VALUES_PARAM_ID) {
//                 normalise_q_values = (bool) param_val;
//             } else if (param_id == DEFAULT_Q_VALUE_PARAM_ID) {
//                 default_q_value = param_val;
//             } else if (param_id == EPSILON_PARAM_ID) {
//                 epsilon = param_val;
//             } else if (param_id == TEMP_PARAM_ID) {
//                 temp = param_val;
//             } else if (param_id == DECAY_FN_PARAM_ID) {
//                 decay_fn = (int) param_val;
//             } else if (param_id == DECAY_FN_SCALE_PARAM_ID) {
//                 decay_fn_scale = param_val;
//             } else if (param_id == ENTROPY_COEFF_PARAM_ID) {
//                 entropy_coeff = param_val;
//             } else if (param_id == ENTROPY_DECAY_FN_PARAM_ID) {
//                 entropy_decay_fn = (int) param_val;
//             } else if (param_id == ENTROPY_DECAY_FN_SCALE_PARAM_ID) {
//                 entropy_decay_fn_scale = param_val;
//             } else {
//                 throw runtime_error("RunID::RunID: Unknown param id: " + param_id);
//             }
//         }
//     }

//     /**
//      * Create thts manager
//     */
//     shared_ptr<ThtsManager> RunID::get_thts_manager(shared_ptr<ThtsEnv> env) 
//     {
//         if (alg_id == UCT_ALG_ID || alg_id == MAX_UCT_ALG_ID) {
//             UctManagerArgs manager_args(env);
//             manager_args.num_threads = num_threads;
//             manager_args.num_envs = num_envs;
//             manager_args.max_depth = max_trial_length;

//             manager_args.mcts_mode = mcts_mode;
//             manager_args.heuristic_fn = mcts_mode ? thts::helper::rollout_heuristic_fn : thts::helper::zero_heuristic_fn;

//             manager_args.adaptive_bias = adaptive_bias;
//             manager_args.bias = bias;

//             return make_shared<UctManager>(manager_args);
//         }

//         if (alg_id == MENTS_ALG_ID || alg_id == RENTS_ALG_ID || alg_id == TENTS_ALG_ID) {
//             MentsManagerArgs manager_args(env);
//             manager_args.num_threads = num_threads;
//             manager_args.num_envs = num_envs;
//             manager_args.max_depth = max_trial_length;

//             manager_args.mcts_mode = mcts_mode;
//             manager_args.heuristic_fn = mcts_mode ? thts::helper::rollout_heuristic_fn : thts::helper::zero_heuristic_fn;

//             manager_args.normalise_q_values = normalise_q_values;
//             manager_args.default_q_value = default_q_value;
//             manager_args.epsilon = epsilon;
            
//             manager_args.temp = temp;

//             return make_shared<MentsManager>(manager_args);
//         }

//         if (alg_id == BTS_ALG_ID) {
//             DentsManagerArgs manager_args(env);
//             manager_args.num_threads = num_threads;
//             manager_args.num_envs = num_envs;
//             manager_args.max_depth = max_trial_length;

//             manager_args.mcts_mode = mcts_mode;
//             manager_args.heuristic_fn = mcts_mode ? thts::helper::rollout_heuristic_fn : thts::helper::zero_heuristic_fn;

//             manager_args.normalise_q_values = normalise_q_values;
//             manager_args.default_q_value = default_q_value;
//             manager_args.epsilon = epsilon;

//             // alpha
//             manager_args.temp = temp;
//             manager_args.temp_decay_fn = nullptr; 
//             if (decay_fn == DECAY_FN_INV_SQRT) {
//                 manager_args.temp_decay_fn = decayed_temp_inv_sqrt;
//             } else if (decay_fn == DECAY_FN_INV_LOG) {
//                 manager_args.temp_decay_fn = decayed_temp_inv_log;
//             }
//             manager_args.temp_decay_fn_x_scale = decay_fn_scale;
            
//             return make_shared<DentsManager>(manager_args);
//         }

//         if (alg_id == DENTS_ALG_ID) {
//             DentsManagerArgs manager_args(env);
//             manager_args.num_threads = num_threads;
//             manager_args.num_envs = num_envs;
//             manager_args.max_depth = max_trial_length;

//             manager_args.mcts_mode = mcts_mode;
//             manager_args.heuristic_fn = mcts_mode ? thts::helper::rollout_heuristic_fn : thts::helper::zero_heuristic_fn;

//             manager_args.normalise_q_values = normalise_q_values;
//             manager_args.default_q_value = default_q_value;
//             manager_args.epsilon = epsilon;

//             // alpha
//             manager_args.temp = temp;
//             manager_args.temp_decay_fn = nullptr; 
//             if (decay_fn == DECAY_FN_INV_SQRT) {
//                 manager_args.temp_decay_fn = decayed_temp_inv_sqrt;
//             } else if (decay_fn == DECAY_FN_INV_LOG) {
//                 manager_args.temp_decay_fn = decayed_temp_inv_log;
//             }
//             manager_args.temp_decay_fn_x_scale = decay_fn_scale;

//             // beta
//             manager_args.entropy_temp = entropy_coeff;
//             manager_args.entropy_temp_decay_fn = nullptr;
//             if (entropy_decay_fn == DECAY_FN_INV_SQRT) {
//                 manager_args.entropy_temp_decay_fn = decayed_temp_inv_sqrt;
//             } else if (entropy_decay_fn == DECAY_FN_INV_LOG) {
//                 manager_args.entropy_temp_decay_fn = decayed_temp_inv_log;
//             }
//             manager_args.entropy_temp_decay_fn_x_scale = entropy_decay_fn_scale;
            
//             return make_shared<DentsManager>(manager_args);
//         }

//         if (alg_id == HMCTS_ALG_ID) {
//             HmctsManagerArgs manager_args(env);
//             manager_args.num_threads = num_threads;
//             manager_args.num_envs = num_envs;
//             manager_args.max_depth = max_trial_length;

//             manager_args.mcts_mode = mcts_mode;
//             manager_args.heuristic_fn = mcts_mode ? thts::helper::rollout_heuristic_fn : thts::helper::zero_heuristic_fn;

//             manager_args.adaptive_bias = adaptive_bias;
//             manager_args.bias = bias;

//             manager_args.total_budget = search_runtime; // search_runtime in units of #trials
//             manager_args.uct_budget_threshold = hmcts_uct_budget;

//             return make_shared<HmctsManager>(manager_args);
//         }

//         stringstream ss;
//         ss << "Error in RunID get_thts_manager for alg_id = " << alg_id;
//         throw runtime_error(ss.str());
//     }








































//     /**
//      * Gets a list of RunID objects from a given expr id
//     */
//     shared_ptr<vector<RunID>> get_run_ids_from_expr_id_prefix(string expr_id_prefix) 
//     {   
//         string expr_id = lookup_expr_id_from_prefix(expr_id_prefix);
//         shared_ptr<vector<RunID>> run_ids = make_shared<vector<RunID>>();

//         // TODO: define RunId's for experiments

//         // expr_id: 000_debug 
//         // debug expr id for debugging
//         if (expr_id == DEBUG_EXPR_ID) {
//             string env_id = D_CHAIN_10_ENV_ID;
//             int max_trial_length = ENV_ID_MAX_TRIAL_LEN.at(D_CHAIN_10_ENV_ID);
//             time_t expr_timestamp = std::time(nullptr);
//             bool eval_wrt_time = false;
//             double search_runtime = 100;
//             double eval_delta = 25;
//             int rollouts_per_mc_eval = 5;
//             int num_repeats = 2;
//             int num_threads = 1;
//             int eval_threads = 1;

//             unordered_map<string,double> alg_params =
//             {
//                 {BIAS_PARAM_ID, 4.0},
//                 {TEMP_PARAM_ID, 1.0},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_CONST},
//                 {DECAY_FN_SCALE_PARAM_ID, 1.0},
//                 {ENTROPY_COEFF_PARAM_ID, 1.0},
//                 {ENTROPY_DECAY_FN_PARAM_ID, DECAY_FN_CONST},
//                 {ENTROPY_DECAY_FN_SCALE_PARAM_ID, 1.0},
//                 {EPSILON_PARAM_ID, 0.1},
//             };

//             vector<string> alg_ids = 
//             {
//                 UCT_ALG_ID,
//                 MENTS_ALG_ID,
//                 BTS_ALG_ID,
//                 DENTS_ALG_ID,
//             };

//             for (string alg_id : alg_ids) {
//                 run_ids->push_back(RunID(
//                     env_id,
//                     expr_id,
//                     expr_timestamp,
//                     alg_id,
//                     alg_params,
//                     eval_wrt_time,
//                     search_runtime,
//                     eval_delta,
//                     rollouts_per_mc_eval,
//                     max_trial_length,
//                     num_repeats,
//                     num_threads,
//                     eval_threads
//                 ));
//             }

//             return run_ids;
//         }

//         // ----
//         // expr_id: 100_supp_dchain_temp_vary 
//         // 10-chain vs temp param
//         // ----
//         // expr_id: 101_supp_mod_dchain_temp_vary 
//         // modified 10-chain vs temp param
//         // ----
//         // expr_id: 102_supp_entropy_temp_vary 
//         // entropy trap 10 vs temp param
//         // ----
//         // expr_id: 103_supp_entropy_temp_15_vary 
//         // entropy trap 15 vs temp param
//         // ----
//         if (expr_id == SUPP_100_DCHAIN_10_TEMP_EXPR_ID
//             || expr_id == SUPP_101_MOD_DCHAIN_10_TEMP_EXPR_ID
//             || expr_id == SUPP_102_ENTROPY_TRAP_10_TEMP_EXPR_ID
//             || expr_id == SUPP_103_ENTROPY_TRAP_15_TEMP_EXPR_ID) 
//         {
//             string env_id = D_CHAIN_10_ENV_ID;
//             if (expr_id == SUPP_101_MOD_DCHAIN_10_TEMP_EXPR_ID) {
//                 env_id = MOD_D_CHAIN_10_ENV_ID;
//             } else if (expr_id == SUPP_102_ENTROPY_TRAP_10_TEMP_EXPR_ID) {
//                 env_id = ENTROPY_TRAP_10_ENV_ID;
//             } else if (expr_id == SUPP_103_ENTROPY_TRAP_15_TEMP_EXPR_ID) {
//                 env_id = ENTROPY_TRAP_15_ENV_ID;
//             }
//             int max_trial_length = ENV_ID_MAX_TRIAL_LEN.at(env_id);
//             time_t expr_timestamp = std::time(nullptr);
//             bool eval_wrt_time = false;
//             double search_runtime = (expr_id == SUPP_103_ENTROPY_TRAP_15_TEMP_EXPR_ID) ? 100000 : 5000;
//             double eval_delta = 50;
//             int rollouts_per_mc_eval = 1; // det env
//             int num_repeats = 25;
//             int num_threads = 8;
//             int eval_threads = 1; // det env

//             // UCT run ids 
//             vector<double> biases_to_try = {
//                 0.001,
//                 0.01,
//                 0.1,
//                 1.0,
//                 10.0,
//                 100.0,
//             };

//             for (double bias : biases_to_try) {
//                 unordered_map<string,double> alg_params =
//                 {
//                     {BIAS_PARAM_ID, bias},
//                 };
//                 run_ids->push_back(RunID(
//                     env_id,
//                     expr_id,
//                     expr_timestamp,
//                     UCT_ALG_ID,
//                     alg_params,
//                     eval_wrt_time,
//                     search_runtime,
//                     eval_delta,
//                     rollouts_per_mc_eval,
//                     max_trial_length,
//                     num_repeats,
//                     num_threads,
//                     eval_threads
//                 ));
//             }
            
//             // MENTS/DENTS/BTS run ids
//             vector<double> temps_to_try = {
//                 0.001,
//                 0.0018,
//                 0.0032,
//                 0.0058,
//                 0.01,
//                 0.018,
//                 0.032,
//                 0.058,
//                 0.1,
//                 0.18,
//                 0.32,
//                 0.58,
//                 1.0,
//                 1.8,
//                 3.2,
//                 5.8,
//                 10.0,
//                 18.0,
//                 32.0,
//                 58.0,
//                 100.0,
//             };
//             vector<string> alg_ids = 
//             {
//                 MENTS_ALG_ID,
//                 BTS_ALG_ID,
//                 DENTS_ALG_ID,
//             };

//             for (double temp : temps_to_try) {
//                 unordered_map<string,double> alg_params =
//                 {
//                     {TEMP_PARAM_ID, temp},
//                     {DECAY_FN_PARAM_ID, DECAY_FN_CONST},
//                     {DECAY_FN_SCALE_PARAM_ID, 1.0},
//                     {ENTROPY_COEFF_PARAM_ID, temp},
//                     {ENTROPY_DECAY_FN_PARAM_ID, DECAY_FN_CONST},
//                     {ENTROPY_DECAY_FN_SCALE_PARAM_ID, 1.0},
//                     {EPSILON_PARAM_ID, 0.01},
//                 };
//                 for (string alg_id : alg_ids) {
//                     run_ids->push_back(RunID(
//                         env_id,
//                         expr_id,
//                         expr_timestamp,
//                         alg_id,
//                         alg_params,
//                         eval_wrt_time,
//                         search_runtime,
//                         eval_delta,
//                         rollouts_per_mc_eval,
//                         max_trial_length,
//                         num_repeats,
//                         num_threads,
//                         eval_threads
//                     ));
//                 }
//             }
            
//             return run_ids;
//         }

//         // ----
//         // expr_id: 110_uct_on_fl_dense / 111_uct_on_fl_sparse_len / 112_uct_on_fl_sparse_discounted 
//         // sanity check UCT on frozen lake stuff??
//         // ----
//         if (expr_id == SUPP_110_UCT_ON_FL_DENSE
//             || expr_id == SUPP_111_UCT_ON_FL_SPARSE_LEN
//             || expr_id == SUPP_112_UCT_ON_FL_SPARSE_DISCOUNTED) 
//         {
//             double default_q_value = -50;
//             string env_id = FROZEN_LAKE_NO_HOLE_DENSE_ENV_ID;
//             if (expr_id == SUPP_111_UCT_ON_FL_SPARSE_LEN) {
//                 env_id = FROZEN_LAKE_NO_HOLE_SPARSE_LEN_ENV_ID;
//             } else if (expr_id == SUPP_112_UCT_ON_FL_SPARSE_DISCOUNTED) {
//                 env_id = FROZEN_LAKE_NO_HOLE_SPARSE_DISCOUNTED_ENV_ID;
//                 default_q_value = 0;
//             }
//             int max_trial_length = ENV_ID_MAX_TRIAL_LEN.at(env_id);
//             time_t expr_timestamp = std::time(nullptr);
//             bool eval_wrt_time = false;
//             double search_runtime = 5000;
//             double eval_delta = 50;
//             int rollouts_per_mc_eval = 1; // det env
//             int num_repeats = 25;
//             int num_threads = 8;
//             int eval_threads = 1; // det env
            
//             // UCT run ids 
//             vector<double> biases_to_try = {
//                 // UctManagerArgs::bias_default,
//                 0.001,
//                 0.01,
//                 0.1,
//                 0.18,
//                 0.32,
//                 0.58,
//                 1.0,
//                 1.8,
//                 3.2,
//                 5.8,
//                 10.0,
//                 18.0,
//                 32.0,
//                 58.0,
//                 100.0,
//                 1000.0,
//                 10000.0,
//             };

//             for (double bias : biases_to_try) {
//                 unordered_map<string,double> alg_params =
//                 {
//                     {BIAS_PARAM_ID, bias},
//                 };
//                 run_ids->push_back(RunID(
//                     env_id,
//                     expr_id,
//                     expr_timestamp,
//                     UCT_ALG_ID,
//                     alg_params,
//                     eval_wrt_time,
//                     search_runtime,
//                     eval_delta,
//                     rollouts_per_mc_eval,
//                     max_trial_length,
//                     num_repeats,
//                     num_threads,
//                     eval_threads
//                 ));
//             }
            
//             // MENTS/DENTS/BTS run ids 
//             vector<double> temps_to_try = {
//                 0.001,
//                 0.01,
//                 0.018,
//                 0.032,
//                 0.058,
//                 0.1,
//                 0.18,
//                 0.32,
//                 0.58,
//                 1.0,
//                 1.8,
//                 3.2,
//                 5.8,
//                 10.0,
//                 18.0,
//                 32.0,
//                 58.0,
//                 100.0,
//                 1000.0,
//                 10000.0,
//             };
//             vector<string> alg_ids = 
//             {
//                 MENTS_ALG_ID,
//                 BTS_ALG_ID,
//                 DENTS_ALG_ID,
//             };

//             for (double temp : temps_to_try) {
//                 unordered_map<string,double> alg_params =
//                 {
//                     {TEMP_PARAM_ID, temp},
//                     {DECAY_FN_PARAM_ID, DECAY_FN_CONST},
//                     {DECAY_FN_SCALE_PARAM_ID, 1.0},
//                     {ENTROPY_COEFF_PARAM_ID, temp},
//                     {ENTROPY_DECAY_FN_PARAM_ID, DECAY_FN_CONST},
//                     {ENTROPY_DECAY_FN_SCALE_PARAM_ID, 1.0},
//                     {EPSILON_PARAM_ID, 0.01},
//                     {DEFAULT_Q_VALUE_PARAM_ID, default_q_value}
//                 };
//                 for (string alg_id : alg_ids) {
//                     run_ids->push_back(RunID(
//                         env_id,
//                         expr_id,
//                         expr_timestamp,
//                         alg_id,
//                         alg_params,
//                         eval_wrt_time,
//                         search_runtime,
//                         eval_delta,
//                         rollouts_per_mc_eval,
//                         max_trial_length,
//                         num_repeats,
//                         num_threads,
//                         eval_threads
//                     ));
//                 }
//             }
            
//             return run_ids;
//         }




//         // ----
//         // expr_id: 80x - frozen lake, deterministic, dense reward
//         // ----
//         if (expr_id == EVAL_FL_D_8x8_EXPR_ID || expr_id == EVAL_FL_D_8x16_EXPR_ID || expr_id == EVAL_FL_D_16x16_EXPR_ID)
//         {
//             // Env params
//             string env_id = FROZEN_LAKE_D_8x8_ENV_ID;
//             if (expr_id == EVAL_FL_D_8x16_EXPR_ID) {
//                 env_id = FROZEN_LAKE_D_8x16_ENV_ID;
//             } else if (expr_id == EVAL_FL_D_16x16_EXPR_ID) {
//                 env_id = FROZEN_LAKE_D_16x16_ENV_ID;
//             }
//             double default_q_value = - (double) ENV_ID_MAX_TRIAL_LEN.at(env_id);
//             int max_trial_length = ENV_ID_MAX_TRIAL_LEN.at(env_id);
//             time_t expr_timestamp = std::time(nullptr);
//             bool eval_wrt_time = false;
//             double search_runtime = 250000;
//             double eval_delta = 500;
//             int rollouts_per_mc_eval = 1024;
//             int num_repeats = 25;
//             int num_threads = 16;
//             int eval_threads = 16;

//             string alg_id;
//             unordered_map<string,double> alg_params;

//             // UCT 
//             // -20.8
//             alg_id = UCT_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 1},
//                 {BIAS_PARAM_ID, 0.01},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // MaxUCT 
//             // -20.25
//             alg_id = MAX_UCT_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 1},
//                 {BIAS_PARAM_ID, 1.92},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // HMCTS
//             // -21.3    
//             alg_id = HMCTS_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 1},
//                 {BIAS_PARAM_ID, 0.0085},
//                 {UCT_BUDGET_PARAM_ID, 4999},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // MENTS
//             // -18.5
//             alg_id = MENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.0038},
//                 {EPSILON_PARAM_ID, 1.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // BTS
//             // -19.36
//             alg_id = BTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 1},
//                 {TEMP_PARAM_ID, 72.9},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_INV_SQRT},
//                 {DECAY_FN_SCALE_PARAM_ID, 100.0},
//                 {EPSILON_PARAM_ID, 1.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // DENTS
//             // -17.2
//             alg_id = DENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 98.6}, // 
//                 {DECAY_FN_PARAM_ID, DECAY_FN_INV_SQRT},
//                 {DECAY_FN_SCALE_PARAM_ID, 100.0},
//                 {ENTROPY_COEFF_PARAM_ID, 0.032}, //
//                 {ENTROPY_DECAY_FN_PARAM_ID, DECAY_FN_CONST},
//                 {ENTROPY_DECAY_FN_SCALE_PARAM_ID, 0.01},
//                 {EPSILON_PARAM_ID, 1.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // RENTS
//             // -18.1
//             alg_id = RENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.24},
//                 {EPSILON_PARAM_ID, 1.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));
            
//             // TENTS
//             // -18.8
//             alg_id = TENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.077},
//                 {EPSILON_PARAM_ID, 1.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             return run_ids;
//         }




//         // ----
//         // expr_id: 81x - frozen lake, deterministic, sparse reward
//         // ----
//         if (expr_id == EVAL_FL_S_8x8_EXPR_ID || expr_id == EVAL_FL_S_8x16_EXPR_ID || expr_id == EVAL_FL_S_16x16_EXPR_ID)
//         {
//             // Env params
//             string env_id = FROZEN_LAKE_S_8x8_ENV_ID;
//             if (expr_id == EVAL_FL_S_8x16_EXPR_ID) {
//                 env_id = FROZEN_LAKE_S_8x16_ENV_ID;
//             } else if (expr_id == EVAL_FL_S_16x16_EXPR_ID) {
//                 env_id = FROZEN_LAKE_S_16x16_ENV_ID;
//             }
//             double default_q_value = 0.0;
//             int max_trial_length = ENV_ID_MAX_TRIAL_LEN.at(env_id);
//             time_t expr_timestamp = std::time(nullptr);
//             bool eval_wrt_time = false;
//             double search_runtime = 250000;
//             double eval_delta = 500;
//             int rollouts_per_mc_eval = 1024;
//             int num_repeats = 25;
//             int num_threads = 16;
//             int eval_threads = 16;

//             string alg_id;
//             unordered_map<string,double> alg_params;

//             // UCT 
//             // 0.820304
//             alg_id = UCT_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 1},
//                 {BIAS_PARAM_ID, 2.44},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // MaxUCT 
//             // 0.808782
//             alg_id = MAX_UCT_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 1},
//                 {BIAS_PARAM_ID, 1.36},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // HMCTS
//             // 0.809554
//             alg_id = HMCTS_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 1},
//                 {BIAS_PARAM_ID, 2.75426},
//                 {UCT_BUDGET_PARAM_ID, 4995},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // MENTS
//             // 0.833115
//             alg_id = MENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.0022},
//                 {EPSILON_PARAM_ID, 1.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));
            
//             // BTS
//             // 0.833384
//             alg_id = BTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 1},
//                 {TEMP_PARAM_ID, 0.0016},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_INV_SQRT},
//                 {DECAY_FN_SCALE_PARAM_ID, 0.53},
//                 {EPSILON_PARAM_ID, 1.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // DENTS
//             // 0.83561
//             alg_id = DENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 1},
//                 {TEMP_PARAM_ID, 0.0014},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_INV_SQRT},
//                 {DECAY_FN_SCALE_PARAM_ID, 0.013},
//                 {ENTROPY_COEFF_PARAM_ID, 0.37},
//                 {ENTROPY_DECAY_FN_PARAM_ID, DECAY_FN_INV_SQRT},
//                 {ENTROPY_DECAY_FN_SCALE_PARAM_ID, 11.9},
//                 {EPSILON_PARAM_ID, 1.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // RENTS
//             // 0.838962
//             alg_id = RENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.0011},
//                 {EPSILON_PARAM_ID, 1.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // TENTS
//             // 0.831394
//             alg_id = TENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.0084},
//                 {EPSILON_PARAM_ID, 1.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             return run_ids;
//         }




//         // ----
//         // expr_id: 82x - slippy frozen lake, stochastic, dense reward
//         // ----
//         if (expr_id == EVAL_SFL_D_4x4_EXPR_ID || expr_id == EVAL_SFL_D_5x5_EXPR_ID || expr_id == EVAL_SFL_D_6x6_EXPR_ID)
//         {
//             // Env params
//             string env_id = SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID;
//             if (expr_id == EVAL_SFL_D_5x5_EXPR_ID) {
//                 env_id = SLIPPY_FROZEN_LAKE_D_5x5_ENV_ID;
//             } else if (expr_id == EVAL_SFL_D_6x6_EXPR_ID) {
//                 env_id = SLIPPY_FROZEN_LAKE_D_6x6_ENV_ID;
//             }
//             double default_q_value = - (double) ENV_ID_MAX_TRIAL_LEN.at(env_id);
//             int max_trial_length = ENV_ID_MAX_TRIAL_LEN.at(env_id);
//             time_t expr_timestamp = std::time(nullptr);
//             bool eval_wrt_time = false;
//             double search_runtime = 250000;
//             double eval_delta = 500;
//             int rollouts_per_mc_eval = 1024;
//             int num_repeats = 25;
//             int num_threads = 16;
//             int eval_threads = 16;

//             string alg_id;
//             unordered_map<string,double> alg_params;

//             // UCT 
//             // -23.5199
//             alg_id = UCT_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 1},
//                 {BIAS_PARAM_ID, 0.0545607},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // MaxUCT 
//             // -23.5268
//             alg_id = MAX_UCT_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 0},
//                 {BIAS_PARAM_ID, 2.74005},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // HMCTS
//             // -23.4768
//             alg_id = HMCTS_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 0},
//                 {BIAS_PARAM_ID, 0.859055},
//                 {UCT_BUDGET_PARAM_ID, 3},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // MENTS
//             // -23.5393
//             alg_id = MENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 1},
//                 {TEMP_PARAM_ID, 0.001},
//                 {EPSILON_PARAM_ID, 0.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // BTS
//             // -23.5238
//             // TODO
//             alg_id = BTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.036216},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_CONST},
//                 {DECAY_FN_SCALE_PARAM_ID, 99.4302},
//                 {EPSILON_PARAM_ID, 0.996121},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // DENTS
//             // -23.4732
//             // TODO
//             alg_id = DENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.001},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_CONST},
//                 {DECAY_FN_SCALE_PARAM_ID, 100},
//                 {ENTROPY_COEFF_PARAM_ID, 0.00547603},
//                 {ENTROPY_DECAY_FN_PARAM_ID, DECAY_FN_INV_LOG},
//                 {ENTROPY_DECAY_FN_SCALE_PARAM_ID, 0.01},
//                 {EPSILON_PARAM_ID, 0.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // RENTS
//             // -23.559
//             alg_id = RENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.001},
//                 {EPSILON_PARAM_ID, 1.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // TENTS
//             // -23.5146
//             alg_id = TENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 1},
//                 {TEMP_PARAM_ID, 0.00829662},
//                 {EPSILON_PARAM_ID, 0.00019767},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             return run_ids;
//         }




//         // ----
//         // expr_id: 83x - slippy frozen lake, stochastic, sparse reward
//         // ----
//         if (expr_id == EVAL_SFL_S_4x4_EXPR_ID || expr_id == EVAL_SFL_S_5x5_EXPR_ID || expr_id == EVAL_SFL_S_6x6_EXPR_ID)
//         {
//             // Env params
//             string env_id = SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID;
//             if (expr_id == EVAL_SFL_S_5x5_EXPR_ID) {
//                 env_id = SLIPPY_FROZEN_LAKE_S_5x5_ENV_ID;
//             } else if (expr_id == EVAL_SFL_S_6x6_EXPR_ID) {
//                 env_id = SLIPPY_FROZEN_LAKE_S_6x6_ENV_ID;
//             }
//             double default_q_value = 0.0;
//             int max_trial_length = ENV_ID_MAX_TRIAL_LEN.at(env_id);
//             time_t expr_timestamp = std::time(nullptr);
//             bool eval_wrt_time = false;
//             double search_runtime = 250000;
//             double eval_delta = 500;
//             int rollouts_per_mc_eval = 1024;
//             int num_repeats = 25;
//             int num_threads = 16;
//             int eval_threads = 16;

//             string alg_id;
//             unordered_map<string,double> alg_params;

//             // UCT 
//             // 0.0382813
//             alg_id = UCT_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 0},
//                 {BIAS_PARAM_ID, 0.104752},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // MaxUCT 
//             // 0.0392578
//             alg_id = MAX_UCT_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 0},
//                 {BIAS_PARAM_ID, 0.247274},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // HMCTS
//             // 0.0386719
//             alg_id = HMCTS_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 0},
//                 {BIAS_PARAM_ID, 0.0988045},
//                 {UCT_BUDGET_PARAM_ID, 517},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // MENTS
//             // 0.0410156
//             alg_id = MENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.001},
//                 {EPSILON_PARAM_ID, 0.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // BTS
//             // 0.0404297
//             // TODO
//             alg_id = BTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.00414547},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_INV_LOG},
//                 {DECAY_FN_SCALE_PARAM_ID, 0.0101548},
//                 {EPSILON_PARAM_ID, 0.227683},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // DENTS
//             // 0.0396484
//             // TODO
//             alg_id = DENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.0738441},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_INV_LOG},
//                 {DECAY_FN_SCALE_PARAM_ID, 42.4924},
//                 {ENTROPY_COEFF_PARAM_ID, 0.0826038},
//                 {ENTROPY_DECAY_FN_PARAM_ID, DECAY_FN_INV_LOG},
//                 {ENTROPY_DECAY_FN_SCALE_PARAM_ID, 28.191},
//                 {EPSILON_PARAM_ID, 0.710385},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // RENTS
//             // 0.0380859
//             alg_id = RENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.001},
//                 {EPSILON_PARAM_ID, 0.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // TENTS
//             // 0.040625
//             alg_id = TENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 1},
//                 {TEMP_PARAM_ID, 0.001},
//                 {EPSILON_PARAM_ID, 0.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             return run_ids;
//         }




//         // ----
//         // expr_id: 84x - sailing, north
//         // ----
//         if (expr_id == EVAL_SAIL_N_8x8_EXPR_ID || expr_id == EVAL_SAIL_N_8x16_EXPR_ID || expr_id == EVAL_SAIL_N_16x16_EXPR_ID)
//         {
//             // Env params
//             string env_id = SAILING_ENV_NORTH_ID;
//             if (expr_id == EVAL_SAIL_N_8x16_EXPR_ID) {
//                 env_id = SAILING_8x16_ENV_NORTH_ID;
//             } else if (expr_id == EVAL_SAIL_N_16x16_EXPR_ID) {
//                 env_id = SAILING_16x16_ENV_NORTH_ID;
//             }
//             double default_q_value = -5.0 * (double) ENV_ID_MAX_TRIAL_LEN.at(env_id);
//             int max_trial_length = ENV_ID_MAX_TRIAL_LEN.at(env_id);
//             time_t expr_timestamp = std::time(nullptr);
//             bool eval_wrt_time = false;
//             double search_runtime = 250000;
//             double eval_delta = 500;
//             int rollouts_per_mc_eval = 1024;
//             int num_repeats = 25;
//             int num_threads = 16;
//             int eval_threads = 16;

//             string alg_id;
//             unordered_map<string,double> alg_params;

//             // UCT 
//             // -78.484
//             alg_id = UCT_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {ADAPTIVE_BIAS_PARAM_ID, 0},
//                 {BIAS_PARAM_ID, 20.0638},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // MaxUCT 
//             // -80.0736
//             alg_id = MAX_UCT_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {ADAPTIVE_BIAS_PARAM_ID, 1},
//                 {BIAS_PARAM_ID, 0.599484},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // HMCTS
//             // -154 (but failed)
//             // TODO
//             alg_id = HMCTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {ADAPTIVE_BIAS_PARAM_ID, 1},
//                 {BIAS_PARAM_ID, 0.599484},
//                 {UCT_BUDGET_PARAM_ID, 1},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // MENTS
//             // -187.941
//             alg_id = MENTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 7.0341},
//                 {EPSILON_PARAM_ID, 0.000876699},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // BTS
//             // -181.67
//             // TODO
//             alg_id = BTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 21.6892},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_INV_LOG},
//                 {DECAY_FN_SCALE_PARAM_ID, 0.0181898},
//                 {EPSILON_PARAM_ID, 0.956878},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // DENTS
//             // -184.881
//             // TODO
//             alg_id = DENTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 4.40294},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_CONST},
//                 {DECAY_FN_SCALE_PARAM_ID, 0.0100076},
//                 {ENTROPY_COEFF_PARAM_ID, 0.0142321},
//                 {ENTROPY_DECAY_FN_PARAM_ID, DECAY_FN_CONST},
//                 {ENTROPY_DECAY_FN_SCALE_PARAM_ID, 99.9819},
//                 {EPSILON_PARAM_ID, 0.999999},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // RENTS
//             // -36.1584
//             alg_id = RENTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 20.0744},
//                 {EPSILON_PARAM_ID, 0.998703},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // TENTS
//             // -186.15
//             alg_id = TENTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {NORMALISE_Q_VALUES_PARAM_ID, 1},
//                 {TEMP_PARAM_ID, 25.4172},
//                 {EPSILON_PARAM_ID, 0.0590783},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             return run_ids;
//         }




//         // ----
//         // expr_id: 84x - sailing, south east
//         // ----
//         if (expr_id == EVAL_SAIL_SE_8x8_EXPR_ID || expr_id == EVAL_SAIL_SE_8x16_EXPR_ID || expr_id == EVAL_SAIL_SE_16x16_EXPR_ID)
//         {
//             // Env params
//             string env_id = SAILING_ENV_SOUTH_EAST_ID;
//             if (expr_id == EVAL_SAIL_SE_8x16_EXPR_ID) {
//                 env_id = SAILING_8x16_ENV_SOUTH_EAST_ID;
//             } else if (expr_id == EVAL_SAIL_SE_16x16_EXPR_ID) {
//                 env_id = SAILING_16x16_ENV_SOUTH_EAST_ID;
//             }
//             double default_q_value = -5.0 * (double) ENV_ID_MAX_TRIAL_LEN.at(env_id);
//             int max_trial_length = ENV_ID_MAX_TRIAL_LEN.at(env_id);
//             time_t expr_timestamp = std::time(nullptr);
//             bool eval_wrt_time = false;
//             double search_runtime = 250000;
//             double eval_delta = 500;
//             int rollouts_per_mc_eval = 1024;
//             int num_repeats = 25;
//             int num_threads = 16;
//             int eval_threads = 16;

//             string alg_id;
//             unordered_map<string,double> alg_params;

//             // UCT 
//             // -98.3836
//             alg_id = UCT_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {ADAPTIVE_BIAS_PARAM_ID, 0},
//                 {BIAS_PARAM_ID, 20.9906},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // MaxUCT 
//             // -90.8139
//             alg_id = MAX_UCT_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {ADAPTIVE_BIAS_PARAM_ID, 1},
//                 {BIAS_PARAM_ID, 1.0},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // HMCTS
//             // -162.195
//             // TODO
//             alg_id = HMCTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {ADAPTIVE_BIAS_PARAM_ID, 0},
//                 {BIAS_PARAM_ID, 52.5813},
//                 {UCT_BUDGET_PARAM_ID, 4990},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // MENTS
//             // -192.017
//             alg_id = MENTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 5.25483},
//                 {EPSILON_PARAM_ID, 0.16076},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // BTS
//             // -194.082
//             // TODO
//             alg_id = BTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 30.0578},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_INV_LOG},
//                 {DECAY_FN_SCALE_PARAM_ID, 0.01},
//                 {EPSILON_PARAM_ID, 1.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // DENTS
//             // -193.007
//             // TODO
//             alg_id = DENTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 6.04038},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_CONST},
//                 {DECAY_FN_SCALE_PARAM_ID, 99.9746},
//                 {ENTROPY_COEFF_PARAM_ID, 0.0960247},
//                 {ENTROPY_DECAY_FN_PARAM_ID, DECAY_FN_INV_LOG},
//                 {ENTROPY_DECAY_FN_SCALE_PARAM_ID, 99.9716},
//                 {EPSILON_PARAM_ID, 0.000167246},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // RENTS
//             // -73.9193
//             alg_id = RENTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 26.8291},
//                 {EPSILON_PARAM_ID, 0.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // TENTS
//             // -196.061
//             alg_id = TENTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {NORMALISE_Q_VALUES_PARAM_ID, 1},
//                 {TEMP_PARAM_ID, 26.721},
//                 {EPSILON_PARAM_ID, 0.0001},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length, 
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             return run_ids;
//         }




//         stringstream ss;
//         ss << "Error in get_run_ids_from_expr_id for expr_id = " << expr_id;
//         throw runtime_error(ss.str());
//     }