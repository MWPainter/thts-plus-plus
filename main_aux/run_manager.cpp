#include "main_aux/run_manager.h"

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
#include "py/py_multiprocessing_thts_env.h"
#include "py/gym_multiprocessing_thts_env.h"

#include "main_aux/envs/d_chain.h"
#include "main_aux/envs/entropy_trap.h"
#include "main_aux/envs/frozen_lake.h"
#include "main_aux/envs/sailing.h"

#include <iomanip>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <string>

using namespace std;
using namespace thts;
using namespace thts::python;

namespace py = pybind11;

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

        if (get_config_value(xpr_config, XPR_OR_ALG_ID_TAG) != XPR_PARAMS_ID_TAG)
        {
            throw runtime_error("In run manager expecting config entry: {XPR_OR_ALG_ID_TAG,XPR_PARAMS_ID_TAG}");
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
    bool RunManager::get_graph_search()         { return get_config_value(xpr_config, XPR_PARAM_ID_GRAPH_SEARCH); }
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
        string env_id = get_env_id();

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
     * Creates the manager and sets algorithm level parameters
     * Then adds the experiment level parameters and returns
    */
    shared_ptr<ThtsManager> RunManager::get_thts_manager(shared_ptr<ThtsEnv> env)
    {
        string alg_id = get_alg_id();
        shared_ptr<ThtsManager> thts_manager = nullptr;

        if (alg_id == ALG_ID_UCT || alg_id == ALG_ID_MAX_UCT) 
        {
            UctManagerArgs manager_args(env);
            manager_args.bias = get_bias();
            thts_manager = make_shared<UctManager>(manager_args);
        }

        else if (alg_id == ALG_ID_HMCTS)
        {
            HmctsManagerArgs manager_args(env);
            manager_args.bias = get_bias();
            manager_args.uct_budget_threshold = get_uct_budget();
            thts_manager = make_shared<HmctsManager>(manager_args);
        }

        else if (alg_id == ALG_ID_MENTS || alg_id == ALG_ID_RENTS || alg_id == ALG_ID_TENTS)
        {
            MentsManagerArgs manager_args(env);
            manager_args.temp_schedule_ptr = make_shared<SqrtSchedule>(get_init_temp(), get_temp_decay_rate());
            manager_args.epsilon = get_epsilon();
            manager_args.default_q_value = get_default_q_value();
            thts_manager = make_shared<MentsManager>(manager_args);
        }

        else if (alg_id == ALG_ID_BTS || alg_id == ALG_ID_DENTS)
        {
            DentsManagerArgs manager_args(env);
            manager_args.temp_schedule_ptr = make_shared<SqrtSchedule>(get_init_temp(), get_temp_decay_rate());
            manager_args.epsilon = get_epsilon();
            manager_args.default_q_value = get_default_q_value();
            
            if (alg_id == ALG_ID_DENTS)
            {
                manager_args.entropy_coeff_schedule_ptr = make_shared<LinearSchedule>(get_init_entropy_coeff(), get_entropy_zero_at());
            }

            thts_manager = make_shared<DentsManager>(manager_args);
        }
        
        if (thts_manager == nullptr)
        {
            stringstream ss;
            ss << "Error in RunManager get_thts_manager for alg_id = " << alg_id;
            throw runtime_error(ss.str());
        }

        thts_manager->num_threads = get_num_search_threads();
        thts_manager->num_envs = std::max(get_num_search_threads(), get_num_eval_threads());
        thts_manager->max_depth = get_max_trial_length();
        thts_manager->heuristic_fn = (get_mcts_mode()) ? helper::rollout_heuristic_fn : helper::zero_heuristic_fn;
        thts_manager->mcts_mode = get_mcts_mode();
        thts_manager->graph_search = get_graph_search();
        thts_manager->first_visit = true;

        return thts_manager;
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


    /**
     * Helper to make a string of:
     * "param1=val1/param2=val2/.../paramN=valN/"
     * Old version output:
     * "param1=val1,param2=val2,...,paramN=valN",
     * but lead to filenames that were too long
    */
    string get_params_string_helper(RunManager& run_manager) 
    {
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
    string RunManager::get_eval_logs_dir() 
    {
        stringstream ss;
        ss << "eval_logs_aux/" 
            << get_xpr_name() << "_" << xpr_timestamp << "/"
            << get_env_id() << "/"
            << get_alg_id() << "/"
            << get_params_string_helper(*this);
        return ss.str();
    }

    /**
     * Int to string with prepended zeros
    */
    string _int_to_string_padded(int num, int pad_size=3) {
        stringstream ss;
        ss << std::setfill('0') << std::setw(pad_size) << num;
        return ss.str();
    }

    std::filesystem::path RunManager::get_eval_log_filename()
    {
        std::filesystem::path dir = get_eval_logs_dir();
        std::filesystem::path filename = dir / "eval_log.txt";

        return filename;
    }
    
    ofstream RunManager::get_eval_log_filestream()
    {
        std::filesystem::path filename = get_eval_log_filename(manager);

        if (!std::filesystem::exists(dir)) {
            fs::create_directories(dir);
        }

        // Open the file (will create it if it doesn’t exist)
        ofstream file(filename, ios::out | ios::trunc);
        if (!file.is_open()) 
        {
            throw runtime_error("Failed to open file: " + filename);
        }

        return file;
    }

    /**
     * Functions for writing to logs files
     */
    void RunManager::write_eval_log_header(std::ofstream& fs)
    {
        // Xpr level params
        fs << "Xpr level params:" << endl << endl;;
        fs << XPR_PARAM_ID_NAME << ","
            << XPR_PARAM_ID_ENV << ","
            << XPR_PARAM_ID_MCTS_MODE << ","
            << XPR_PARAM_ID_GRAPH_SEARCH << ","
            << XPR_PARAM_ID_MAX_TRIAL_LENGTH << ","
            << XPR_PARAM_ID_RUNTIME_BOUNDED << ","
            << XPR_PARAM_ID_TERMINATION_BOUND << ","
            << XPR_PARAM_ID_REPEATED_RUNS_PER_ALG << ","
            << XPR_PARAM_ID_SEARCH_THREADS << ","
            << XPR_PARAM_ID_EVAL_DELTA << ","
            << XPR_PARAM_ID_EVAL_ROLLOUTS << ","
            << XPR_PARAM_ID_EVAL_THREADS << endl;
        fs << get_xpr_name() << ","
            << get_env_id() << ","
            << get_mcts_mode() << ","
            << get_graph_search() << ","
            << get_max_trial_length() << ","
            << xpr_is_runtime_bounded() << ","
            << get_termination_bound() << ","
            << get_repeated_runs_per_alg() << ","
            << get_num_search_threads() << ","
            << get_eval_delta() << ","
            << get_num_eval_rollouts() << ","
            << get_num_eval_threads() << endl;

        // Alg level params
        string alg_id = get_alg_id();
        fs << endl << alg_id << " params: " << endl << endl;
        bool first_iter = true;
        for (string alg_param_id : ALG_ID_TO_ALG_PARAM_IDS[alg_id])
        {
            if (!first_iter)
            {
                fs << ",";
            }
            first_iter = false;
            fs << alg_param_id;
        }
        fs << endl;

        // Header for main body
        fs << endl << "Evals: " << endl << endl;
        results_evals_fs << "run_idx,eval,eval_std,num_trials,runtime,num_eval_samples" << endl;
    }

    void RunManager::write_eval_line(
        ofstream& fs, int run_idx, double eval, double eval_std, int num_trials, double runtime, int num_eval_samples)
    {
        fs << run_idx << "," << eval << "," << eval_std << "," << num_trials << "," << runtime << "," << num_eval_samples << endl;
    }


    std::filesystem::path RunManager::get_tree_log_filename(int run_idx)
    {

        stringstream filename_ss;
        filename_ss << "tree_log_run_"  <<_int_to_padded_string(run_idx) << ".txt";

        std::filesystem::path dir = get_eval_logs_dir();
        std::filesystem::path filename = dir / filename_ss.str();

        return filename;
    }

    std::ofstream RunManager::get_tree_log_filestream(int run_idx)
    {

        std::filesystem::path filename = get_eval_log_filename(manager, run_idx);

        if (!std::filesystem::exists(dir)) {
            fs::create_directories(dir);
        }

        // Open the file (will create it if it doesn’t exist)
        ofstream file(filename, ios::out | ios::trunc);
        if (!file.is_open()) 
        {
            throw runtime_error("Failed to open file: " + filename);
        }

        return file;
    }

    void RunManager::dump_tree_log(shared_ptr<ThtsDNode> root_node, int run_idx)
    {
        ofstream tree_log_fs = get_tree_log_filestream(run_idx);
        tree_log_fs << root_node->get_pretty_print_string(3) << endl;
        tree_log_fs.close()
    }
}
