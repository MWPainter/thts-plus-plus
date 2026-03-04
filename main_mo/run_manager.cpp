#include "main_mo/run_manager.h"

#include "mo/algorithms/chmcts/ch_czt_manager.h"
#include "mo/algorithms/chmcts/ch_bts_manager.h"
#include "mo/algorithms/chmcts/ch_uct_manager.h"
#include "mo/algorithms/contextual_zooming/czt_manager.h"
#include "mo/algorithms/prior/ch_hvuct_manager.h"
#include "mo/algorithms/prior/ch_pareto_uct_manager.h"
#include "mo/algorithms/prior/ch_cheby_manager.h"

#include "mo/algorithms/chmcts/ch_czt_decision_node.h"
#include "mo/algorithms/chmcts/ch_bts_decision_node.h"
#include "mo/algorithms/chmcts/ch_uct_decision_node.h"
#include "mo/algorithms/contextual_zooming/czt_decision_node.h"
#include "mo/algorithms/prior/ch_hvuct_decision_node.h"
#include "mo/algorithms/prior/ch_pareto_uct_decision_node.h"
#include "mo/algorithms/prior/ch_cheby_decision_node.h"

#include "algorithms/common/decaying_temp.h"

#include "py/pickle_wrapper.h"
#include "py/mo_py_multiprocessing_thts_env.h"
#include "py/mo_gym_multiprocessing_thts_env.h"
#include "py/timed_mo_gym_multiprocessing_thts_env.h"

#include "main_mo/envs/tree_env.h"
#include "main_mo/envs/test_mo_thts_env.h"
#include "main_mo/envs/ported_dst.h"
#include "main_mo/envs/ported_resource_gathering.h"

#include <iomanip>
#include <iostream>
#include <set>
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
    RunManager::RunManager(
        std::time_t xpr_timestamp, 
        ConfigMap xpr_config, 
        ConfigMap alg_config,
        std::string xpr_dir_override,
        std::string thts_unique_filename) :
            xpr_timestamp(xpr_timestamp), 
            xpr_config(xpr_config), 
            alg_config(alg_config),
            xpr_dir_override(xpr_dir_override),
            thts_unique_filename(thts_unique_filename)
    {
        validate_config_or_raise_exception();
    }

    /**
     *  Checks all expected params are present, and that no additional params
     */  
    void RunManager::validate_config_or_raise_exception()
    {
        if (xpr_config.size() != 20)
        {
            throw runtime_error("Expecting 20 entries in the xpr level config.");
        }

        if (get_config_value<std::string>(xpr_config, XPR_OR_ALG_ID_TAG) != XPR_PARAMS_ID_TAG)
        {
            throw runtime_error("In run manager expecting config entry: {XPR_OR_ALG_ID_TAG,XPR_PARAMS_ID_TAG}");
        }

        vector<string> xpr_param_ids = 
        {
            XPR_PARAM_ID_NAME, 
            XPR_PARAM_ID_ENV, 
            XPR_PARAM_ID_ENV_SIZE,
            XPR_PARAM_ID_MCTS_MODE, 
            XPR_PARAM_ID_MAX_TRIAL_LENGTH,
            XPR_PARAM_ID_GRAPH_SEARCH,
            XPR_PARAM_ID_VECTOR_VISIT_COUNTS,
            XPR_PARAM_ID_RUNTIME_BOUNDED, 
            XPR_PARAM_ID_TERMINATION_BOUND, 
            XPR_PARAM_ID_REPEATED_RUNS_PER_ALG, 
            XPR_PARAM_ID_SEARCH_THREADS, 
            XPR_PARAM_ID_EVAL_DELTA, 
            XPR_PARAM_ID_EVAL_ROLLOUTS, 
            XPR_PARAM_ID_EVAL_THREADS,
            XPR_PARAM_ID_CONVEX_HULL_MAX_SIZE,
            XPR_PARAM_ID_CONVEX_HULL_TOLERANCE,
            XPR_PARAM_ID_USE_SOLVED_LABELLING,
            XPR_PARAM_ID_SOLVED_LABELLING_FAIL_CONFIDENCE,
            XPR_PARAM_ID_SOLVED_LABELLING_TOLERANCE,
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

        string alg_id = get_config_value<std::string>(alg_config, XPR_OR_ALG_ID_TAG);

        set<string> alg_ids =
        {
            ALG_ID_CHVI,
            ALG_ID_CZT, 
            ALG_ID_CZT_DOUBLING, 
            ALG_ID_CH_UCT, 
            ALG_ID_CH_CZT, 
            ALG_ID_CH_CZT_DOUBLING, 
            ALG_ID_CH_BTS, 
            ALG_ID_CH_DENTS, 
            ALG_ID_CH_HVUCT, 
            ALG_ID_CH_PARETO, 
            ALG_ID_CH_CHEBY,
            ALG_ID_CH_STANDARD_CHEBY,
        };

        if (!alg_ids.contains(alg_id))
        {
            stringstream ss;
            ss << "Found unrecognised algorithm id " << alg_id << " in alg level config.";
            throw runtime_error(ss.str());
        }

        vector<string> param_ids_expecting = ALG_ID_TO_ALG_PARAM_IDS.at(alg_id);
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
        for (const vector<ConfigMap>& config : ALL_CONFIGS)
        {
            const ConfigMap& xpr_config = config[0];
            if (get_config_value<std::string>(xpr_config, XPR_OR_ALG_ID_TAG) != XPR_PARAMS_ID_TAG)
            {
                throw runtime_error("Expecting first map in each config (vector) to specify xpr level config with correct tagging.");
            }
            if (!xpr_config.contains(XPR_PARAM_ID_NAME)) 
            {
                throw runtime_error("Expecting xpr level config to specify an xpr_name");
            }
        }

        // Lookup
        for (const vector<ConfigMap>& config : ALL_CONFIGS)
        {
            string xpr_name = get_config_value<std::string>(config[0], XPR_PARAM_ID_NAME);
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
    shared_ptr<vector<RunManager>> RunManager::get_run_managers_from_config_vector(
        vector<ConfigMap>& config_vector,
        time_t xpr_timestamp,
        string xpr_dir_override)
    {
        shared_ptr<vector<RunManager>> run_managers = std::make_shared<vector<RunManager>>();

        // If env size is variable, then unpack sizes into seperate config_vectors
        bool env_size_is_variable = config_value_is_int_vector(config_vector[0], XPR_PARAM_ID_ENV_SIZE);
        if (env_size_is_variable) {
            vector<int> env_sizes = get_config_value<vector<int>>(config_vector[0], XPR_PARAM_ID_ENV_SIZE);
            for (int env_size : env_sizes) {
                // Copy config vector
                vector<ConfigMap> new_config_vector = config_vector;

                // Override env size
                ConfigMap xpr_config = new_config_vector[0];
                xpr_config[XPR_PARAM_ID_ENV_SIZE] = env_size;

                // Override max trial length on a per env size basis
                string env_id = get_config_value<std::string>(xpr_config, XPR_PARAM_ID_ENV);
                if (VARIABLE_SIZED_VAMPLEW_DST_ENVS.contains(env_id) || VARIABLE_SIZED_IMPROVED_DST_ENVS.contains(env_id))
                {
                    int max_xy_distance = get_max_xy_for_width(env_size);
                    xpr_config[XPR_PARAM_ID_MAX_TRIAL_LENGTH] = max_xy_distance * 2;
                }
                else 
                {
                    throw runtime_error("Max trial length not overridden variable sizes env" + env_id);
                }

                // Update xpr_config in config_vector
                new_config_vector[0] = xpr_config;

                // Recursively get run managers for the new config vector
                shared_ptr<vector<RunManager>> new_run_managers = get_run_managers_from_config_vector(new_config_vector, xpr_timestamp, xpr_dir_override);
                run_managers->insert(run_managers->end(), new_run_managers->begin(), new_run_managers->end());
            }
            return run_managers;
        }

        // Otherwise, just return the run managers for the single config
        for (size_t i=1; i<config_vector.size(); i++)
        {  
            run_managers->push_back(RunManager(xpr_timestamp, config_vector[0], config_vector[i], xpr_dir_override));
        }   
        return run_managers;
    }


    /**
     * Getters - xpr level config
     */
    string RunManager::get_xpr_name()           { return get_config_value<std::string>(xpr_config, XPR_PARAM_ID_NAME); }
    string RunManager::get_env_id()             { return get_config_value<std::string>(xpr_config, XPR_PARAM_ID_ENV); }
    int RunManager::get_env_size()              { return get_config_value<int>(xpr_config, XPR_PARAM_ID_ENV_SIZE); }
    bool RunManager::get_mcts_mode()            { return get_config_value<bool>(xpr_config, XPR_PARAM_ID_MCTS_MODE); }
    bool RunManager::get_graph_search()         { return get_config_value<bool>(xpr_config, XPR_PARAM_ID_GRAPH_SEARCH); }
    bool RunManager::get_vector_visit_counts()  { return get_config_value<bool>(xpr_config, XPR_PARAM_ID_VECTOR_VISIT_COUNTS); }
    int RunManager::get_max_trial_length()      { return get_config_value<int>(xpr_config, XPR_PARAM_ID_MAX_TRIAL_LENGTH); }
    bool RunManager::xpr_is_runtime_bounded()   { return get_config_value<bool>(xpr_config, XPR_PARAM_ID_RUNTIME_BOUNDED); }
    double RunManager::get_termination_bound()  { return get_config_value<double>(xpr_config, XPR_PARAM_ID_TERMINATION_BOUND); }
    int RunManager::get_repeated_runs_per_alg() { return get_config_value<int>(xpr_config, XPR_PARAM_ID_REPEATED_RUNS_PER_ALG); }
    int RunManager::get_num_search_threads()    { return get_config_value<int>(xpr_config, XPR_PARAM_ID_SEARCH_THREADS); }
    double RunManager::get_eval_delta()         { return get_config_value<double>(xpr_config, XPR_PARAM_ID_EVAL_DELTA); }
    int RunManager::get_num_eval_rollouts()     { return get_config_value<int>(xpr_config, XPR_PARAM_ID_EVAL_ROLLOUTS); }
    int RunManager::get_num_eval_threads()      { return get_config_value<int>(xpr_config, XPR_PARAM_ID_EVAL_THREADS); }
    int RunManager::get_convex_hull_max_size()  { return get_config_value<int>(xpr_config, XPR_PARAM_ID_CONVEX_HULL_MAX_SIZE); }
    double RunManager::get_convex_hull_tolerance() { return get_config_value<double>(xpr_config, XPR_PARAM_ID_CONVEX_HULL_TOLERANCE); }
    bool RunManager::get_use_solved_labelling() { return get_config_value<bool>(xpr_config, XPR_PARAM_ID_USE_SOLVED_LABELLING); }
    double RunManager::get_solved_labelling_fail_confidence() { return get_config_value<double>(xpr_config, XPR_PARAM_ID_SOLVED_LABELLING_FAIL_CONFIDENCE); }
    double RunManager::get_solved_labelling_tolerance() { return get_config_value<double>(xpr_config, XPR_PARAM_ID_SOLVED_LABELLING_TOLERANCE); }

    /**
     * Getters - alg level config
     */
    string RunManager::get_alg_id()                         { return get_config_value<std::string>(alg_config, XPR_OR_ALG_ID_TAG); }
    bool RunManager::is_chvi()                              { return get_alg_id() == ALG_ID_CHVI; }
    double RunManager::get_bias()                           { return get_config_value<double>(alg_config, ALG_PARAM_ID_BIAS); }
    double RunManager::get_czt_ball_split_visit_thresh()    { return get_config_value<double>(alg_config, ALG_PARAM_ID_CZT_BALL_SPLIT_VISIT_THRESH); }
    double RunManager::get_min_log2_N()                     { return get_config_value<double>(alg_config, ALG_PARAM_ID_MIN_LOG2_N); }
    double RunManager::get_init_temp()                      { return get_config_value<double>(alg_config, ALG_PARAM_ID_INIT_TEMP); }
    double RunManager::get_temp_decay_rate()                { return get_config_value<double>(alg_config, ALG_PARAM_ID_TEMP_DECAY_RATE); }
    double RunManager::get_init_entropy_coeff()             { return get_config_value<double>(alg_config, ALG_PARAM_ID_INIT_ENTROPY_COEFF); }
    double RunManager::get_entropy_zero_at()                { return get_config_value<double>(alg_config, ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT); }
    double RunManager::get_epsilon()                        { return get_config_value<double>(alg_config, ALG_PARAM_ID_EPSILON); }
    double RunManager::get_default_q_value()                { return get_config_value<double>(alg_config, ALG_PARAM_ID_DEFAULT_Q_VALUE); }


    /**
     * Returns if the env we are using is a python env
    */
    bool RunManager::is_python_env()
    {
        string env_id = get_env_id();
        return (PY_ENVS.contains(env_id) 
            || GYM_ENVS.contains(env_id)
            || TIMED_GYM_ENVS.contains(env_id)
            || DST_PY_ENVS.contains(env_id));
    }

    /**
     * Returns an instance of MoThtsEnv to use for this run
    */
    shared_ptr<MoThtsEnv> RunManager::get_env()
    {
        string unique_filename;
        if (!this->thts_unique_filename.empty()) {
            unique_filename = this->thts_unique_filename;
        } else {
            unique_filename = get_eval_logs_dir();
        }
        string env_id = get_env_id();

        if (GYM_ENVS.contains(env_id)) {
            shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
            return make_shared<MoGymMultiprocessingThtsEnv>(pickle_wrapper, unique_filename, env_id);
        }

        if (env_id == ENV_ID_RESOURCE_GATHER_CPP || env_id == ENV_ID_RESOURCE_GATHER_TIMED_CPP)
        {
            bool timed = (env_id == ENV_ID_RESOURCE_GATHER_TIMED_CPP);
            return make_shared<PortedResourceGatheringThtsEnv>(timed);
        }

        if (TIMED_GYM_ENVS.contains(env_id)) {
            shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
            return make_shared<TimedMoGymMultiprocessingThtsEnv>(pickle_wrapper, unique_filename, env_id);
        }

        if (DST_ENVS.contains(env_id)) 
        {
            if (env_id == ENV_ID_VAMPLEW_DST_10_CPP 
                || env_id == ENV_ID_VAMPLEW_STOCH_DST_10_CPP
                || env_id == ENV_ID_VAMPLEW_DST_VARIABLE_CPP 
                || env_id == ENV_ID_VAMPLEW_STOCH_DST_VARIABLE_CPP)
            {
                bool swept_by_current = (env_id == ENV_ID_VAMPLEW_STOCH_DST_10_CPP);
                double swept_by_current_prob = swept_by_current ? 0.2 : 0.0;
                int max_timestep = this->get_max_trial_length();
                int map_id = this->get_env_size();
                if (env_id == ENV_ID_VAMPLEW_DST_10_CPP || env_id == ENV_ID_VAMPLEW_STOCH_DST_10_CPP)
                {
                    map_id = 10;
                }
                return make_shared<PortedDeepSeaTreasureThtsEnv>(map_id, swept_by_current_prob, max_timestep);
            }

            if (env_id == ENV_ID_IMPROVED_DST_VARIABLE_CPP || env_id == ENV_ID_IMPROVED_STOCH_DST_VARIABLE_CPP)
            {
                throw runtime_error("Improved DST not ported yet");
                // int map_id = this->get_env_size();
                // bool swept_by_current = (env_id == ENV_ID_IMPROVED_STOCH_DST_VARIABLE_CPP);
                // double swept_by_current_prob = swept_by_current ? 0.2 : 0.0;
                // int max_timestep = this->get_max_trial_length();
                // return make_shared<PortedDeepSeaTreasureThtsEnv>(map_id, swept_by_current_prob, max_timestep);
            }

            py::gil_scoped_acquire acq;
    
            bool swept_by_current = STOCH_PY_DST_ENVS.contains(env_id);
            double swept_by_current_prob = swept_by_current ? 0.2 : 0.0;
            bool is_vamplew = VAMPLEW_PY_DST_ENVS.contains(env_id);
            int map_id = 1;

            if (env_id == ENV_ID_VAMPLEW_DST_10 || env_id == ENV_ID_VAMPLEW_STOCH_DST_10) 
            { 
                map_id = 10; 
            }
            else if (VARIABLE_SIZED_VAMPLEW_DST_ENVS.contains(env_id) || VARIABLE_SIZED_IMPROVED_DST_ENVS.contains(env_id))
            {
                map_id = this->get_env_size();
            }
    
            py::dict kw_args;
            kw_args["swept_by_current_prob"] = to_string(swept_by_current_prob);
            kw_args["is_vamplew"] = is_vamplew ? "True" : "False";
            kw_args["max_steps"] = to_string(this->get_max_trial_length());
            kw_args["map_id"] = to_string(map_id);
            shared_ptr<py::dict> kw_args_ptr = make_shared<py::dict>(kw_args);

            shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
            string module_name = "main_mo.envs.custom_deep_sea_treasure";
            string class_name = "ImprovedDeepSeaTreasureThtsEnv";
            return make_shared<MoPyMultiprocessingThtsEnv>(
                pickle_wrapper, 
                unique_filename, 
                module_name, 
                class_name, 
                kw_args_ptr);
        }

        if (env_id == ENV_ID_FRUIT_TREE_7 
            || env_id == ENV_ID_FRUIT_TREE_STOCH_5 
            || env_id == ENV_ID_FRUIT_TREE_STOCH_7) 
        {
            py::gil_scoped_acquire acq;
            
            py::dict kw_args;
            kw_args["depth"] = to_string((env_id == ENV_ID_FRUIT_TREE_STOCH_5) ? 5 : 7);
            kw_args["action_noise"] = to_string((env_id == ENV_ID_FRUIT_TREE_7) ? 0.0 : 0.2);
            shared_ptr<py::dict> kw_args_ptr = make_shared<py::dict>(kw_args);

            shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
            string module_name = "main_mo.envs.custom_fruit_tree";
            string class_name = "StochFruitTreeThtsEnv";
            return make_shared<MoPyMultiprocessingThtsEnv>(
                pickle_wrapper, 
                unique_filename, 
                module_name, 
                class_name, 
                kw_args_ptr);
        }

        if (env_id == ENV_ID_DEBUG_1) 
        {
            int walk_len = 10;
            double wrong_dir_prob = 0.0;
            bool add_extra_rewards = false;
            return make_shared<TestMoThtsEnv>(walk_len, wrong_dir_prob, add_extra_rewards);
        }
        if (env_id == ENV_ID_DEBUG_2) 
        {
            int walk_len = 10;
            double wrong_dir_prob = 0.25;
            bool add_extra_rewards = false;
            return make_shared<TestMoThtsEnv>(walk_len, wrong_dir_prob, add_extra_rewards);
        }
        if (env_id == ENV_ID_DEBUG_3) 
        {
            int walk_len = 10;
            double wrong_dir_prob = 0.0;
            bool add_extra_rewards = true;
            return make_shared<TestMoThtsEnv>(walk_len, wrong_dir_prob, add_extra_rewards);
        }
        if (env_id == ENV_ID_DEBUG_4) 
        {
            int walk_len = 10;
            double wrong_dir_prob = 0.25;
            bool add_extra_rewards = true;
            return make_shared<TestMoThtsEnv>(walk_len, wrong_dir_prob, add_extra_rewards);
        }

        if (env_id == ENV_ID_PY_DEBUG_1) 
        {
            py::gil_scoped_acquire acq;
            
            py::dict kw_args;
            kw_args["walk_len"] = to_string(10);
            kw_args["wrong_dir_prob"] = to_string(0.0);
            kw_args["add_extra_rewards"] = "False";
            shared_ptr<py::dict> kw_args_ptr = make_shared<py::dict>(kw_args);

            shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
            string module_name = "main_mo.envs.test_mo_thts_env";
            string class_name = "TestMoThtsEnv";
            return make_shared<MoPyMultiprocessingThtsEnv>(
                pickle_wrapper, 
                unique_filename, 
                module_name, 
                class_name, 
                kw_args_ptr);
        }
        if (env_id == ENV_ID_PY_DEBUG_2) 
        {
            py::gil_scoped_acquire acq;
            
            py::dict kw_args;
            kw_args["walk_len"] = to_string(10);
            kw_args["wrong_dir_prob"] = to_string(0.25);
            kw_args["add_extra_rewards"] = "False";
            shared_ptr<py::dict> kw_args_ptr = make_shared<py::dict>(kw_args);

            shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
            string module_name = "main_mo.envs.test_mo_thts_env";
            string class_name = "TestMoThtsEnv";
            return make_shared<MoPyMultiprocessingThtsEnv>(
                pickle_wrapper, 
                unique_filename, 
                module_name, 
                class_name, 
                kw_args_ptr);
        }
        if (env_id == ENV_ID_PY_DEBUG_3) 
        {
            py::gil_scoped_acquire acq;
            
            py::dict kw_args;
            kw_args["walk_len"] = to_string(10);
            kw_args["wrong_dir_prob"] = to_string(0.0);
            kw_args["add_extra_rewards"] = "True";
            shared_ptr<py::dict> kw_args_ptr = make_shared<py::dict>(kw_args);

            shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
            string module_name = "main_mo.envs.test_mo_thts_env";
            string class_name = "TestMoThtsEnv";
            return make_shared<MoPyMultiprocessingThtsEnv>(
                pickle_wrapper, 
                unique_filename, 
                module_name, 
                class_name, 
                kw_args_ptr);
        }
        if (env_id == ENV_ID_PY_DEBUG_4) 
        {
            py::gil_scoped_acquire acq;
            
            py::dict kw_args;
            kw_args["walk_len"] = to_string(10);
            kw_args["wrong_dir_prob"] = to_string(0.25);
            kw_args["add_extra_rewards"] = "True";
            shared_ptr<py::dict> kw_args_ptr = make_shared<py::dict>(kw_args);

            shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
            string module_name = "main_mo.envs.test_mo_thts_env";
            string class_name = "TestMoThtsEnv";
            return make_shared<MoPyMultiprocessingThtsEnv>(
                pickle_wrapper, 
                unique_filename, 
                module_name, 
                class_name, 
                kw_args_ptr);
        }

        if (env_id == ENV_ID_TOY_TREE_DENSE)
        {
            return make_shared<ToyTreeEnv>(2, 5, 10, false);
        }
        if (env_id == ENV_ID_TOY_TREE_SPARSE)
        {
            return make_shared<ToyTreeEnv>(2, 5, 10, true);
        }

        stringstream ss;
        ss << "Error in get_env for env_id = " << env_id;
        throw runtime_error(ss.str());
    }

    Eigen::ArrayXd RunManager::get_env_value_upper_bound()
    {
        
        string env_id = get_env_id();

        if (env_id == ENV_ID_DEBUG_1
            || env_id == ENV_ID_DEBUG_2
            || env_id == ENV_ID_DEBUG_3
            || env_id == ENV_ID_DEBUG_4
            || env_id == ENV_ID_PY_DEBUG_1
            || env_id == ENV_ID_PY_DEBUG_2
            || env_id == ENV_ID_PY_DEBUG_3
            || env_id == ENV_ID_PY_DEBUG_4) 
        {
            double max_steps = 10.0;
            unordered_set<string> four_d_envs = 
            {
                ENV_ID_DEBUG_3,
                ENV_ID_DEBUG_4,
                ENV_ID_PY_DEBUG_3,
                ENV_ID_PY_DEBUG_4,
            };
            Eigen::ArrayXd r_max = Eigen::ArrayXd(2);
            if (four_d_envs.contains(env_id)) {
                r_max = Eigen::ArrayXd(4);
            }
            r_max[0] = 0.0; 
            r_max[1] = 0.0; 
            if (four_d_envs.contains(env_id)) {
                r_max[2] = 1.0;
                r_max[3] = 1.0;
            }
            return max_steps * r_max;
        }

        if (env_id == ENV_ID_TOY_TREE_DENSE
            || env_id == ENV_ID_TOY_TREE_SPARSE)
        {
            double max_steps = 10.0;
            Eigen::ArrayXd r_min = Eigen::ArrayXd::Ones(2);
            return max_steps * r_min;
        }

        if (env_id == ENV_ID_VAMPLEW_DST
            || env_id == ENV_ID_VAMPLEW_STOCH_DST)
        {
            Eigen::ArrayXd max_val = Eigen::ArrayXd::Zero(2);
            max_val[0] = 124.0; 
            max_val[1] = 0.0; 
            return max_val;
        }

        if (env_id == ENV_ID_IMPROVED_DST
            || env_id == ENV_ID_IMPROVED_STOCH_DST)
        {
            Eigen::ArrayXd max_val = Eigen::ArrayXd::Zero(3);
            max_val[0] = 124.0; 
            max_val[1] = 0.0; 
            max_val[2] = 0.0;
            return max_val;
        }

        if (env_id == ENV_ID_VAMPLEW_DST_MO_GYM
            || env_id == ENV_ID_VAMPLEW_STOCH_DST_MO_GYM)
        {
            Eigen::ArrayXd max_val = Eigen::ArrayXd::Zero(2);
            max_val[0] = 23.7; 
            max_val[1] = 0.0;
            return max_val;
        }

        if (env_id == ENV_ID_VAMPLEW_DST_10
            || env_id == ENV_ID_VAMPLEW_STOCH_DST_10)
        {
            Eigen::ArrayXd max_val = Eigen::ArrayXd::Zero(2);
            max_val[0] = 50.0; 
            max_val[1] = 0.0;
            return max_val;
        }

        if (env_id == ENV_ID_VAMPLEW_DST_10_CPP
            || env_id == ENV_ID_VAMPLEW_STOCH_DST_10_CPP)
        {
            Eigen::ArrayXd max_val = Eigen::ArrayXd::Zero(2);
            max_val[0] = 50.0; 
            max_val[1] = 0.0;
            return max_val;
        }

        if (VARIABLE_SIZED_VAMPLEW_DST_ENVS.contains(env_id))
        {
            Eigen::ArrayXd max_val = Eigen::ArrayXd::Zero(2);
            TreasureMap map = *get_map(this->get_env_size());
            double max_treasure = map.back().value;
            max_val[0] = max_treasure; 
            max_val[1] = 0.0;
            return max_val;
        }

        if (VARIABLE_SIZED_IMPROVED_DST_ENVS.contains(env_id))
        {
            Eigen::ArrayXd max_val = Eigen::ArrayXd::Zero(3);
            TreasureMap map = *get_map(this->get_env_size());
            double max_treasure = map.back().value;
            max_val[0] = max_treasure; 
            max_val[1] = 0.0;
            max_val[2] = 0.0;
            return max_val;
        }

        if (env_id == ENV_ID_FRUIT_TREE_7
            || env_id == ENV_ID_FRUIT_TREE_STOCH_5
            || env_id == ENV_ID_FRUIT_TREE_STOCH_7)
        {
            return Eigen::ArrayXd::Ones(6) * 10.0;
        }

        if (env_id == ENV_ID_RESOURCE_GATHER
            || env_id == ENV_ID_RESOURCE_GATHER_TIMED
            || env_id == ENV_ID_RESOURCE_GATHER_CPP
            || env_id == ENV_ID_RESOURCE_GATHER_TIMED_CPP)
        {
            Eigen::ArrayXd max_val = Eigen::ArrayXd(3);
            if (env_id == ENV_ID_RESOURCE_GATHER_TIMED || env_id == ENV_ID_RESOURCE_GATHER_TIMED_CPP)
            {
                max_val = Eigen::ArrayXd(4);
            }
            max_val[0] = 0.0;
            max_val[1] = 1.0;
            max_val[2] = 1.0;
            if (env_id == ENV_ID_RESOURCE_GATHER_TIMED || env_id == ENV_ID_RESOURCE_GATHER_TIMED_CPP) {
                max_val[3] = 0.0;
            }
            return max_val;
        }
        
        if (env_id == ENV_ID_BREAKABLE_BOTTLES) {
            Eigen::ArrayXd max_val = Eigen::ArrayXd(3);
            max_val[0] = 0.0;
            max_val[1] = 50.0;
            max_val[2] = 0.0;
            return max_val;
        }

        if (env_id == ENV_ID_FOUR_ROOM
            || env_id == ENV_ID_FOUR_ROOM_TIMED) 
        {
            Eigen::ArrayXd max_val = Eigen::ArrayXd(3);
            if (env_id == ENV_ID_FOUR_ROOM_TIMED) {
                max_val = Eigen::ArrayXd(4);
            }
            max_val[0] = 4.0;
            max_val[1] = 4.0;
            max_val[2] = 4.0;
            if (env_id == ENV_ID_FOUR_ROOM_TIMED) {
                max_val[3] = 0.0;
            }
            return max_val;
        }

        if (env_id == ENV_ID_MINECART_DETERMINISTIC)
        {
            Eigen::ArrayXd max_val = Eigen::ArrayXd(3);
            max_val[0] = 1.5;
            max_val[1] = 1.5;
            max_val[2] = 0.0;
            return max_val;
        }

        throw runtime_error("get_env_value_upper_bound not implemented for env_id = " + get_env_id());
    }

    Eigen::ArrayXd RunManager::get_env_value_lower_bound()
    {
        string env_id = get_env_id();

        if (env_id == ENV_ID_DEBUG_1
            || env_id == ENV_ID_DEBUG_2
            || env_id == ENV_ID_DEBUG_3
            || env_id == ENV_ID_DEBUG_4
            || env_id == ENV_ID_PY_DEBUG_1
            || env_id == ENV_ID_PY_DEBUG_2
            || env_id == ENV_ID_PY_DEBUG_3
            || env_id == ENV_ID_PY_DEBUG_4) 
        {
            double max_steps = 10.0;
            unordered_set<string> four_d_envs = 
            {
                ENV_ID_DEBUG_3,
                ENV_ID_DEBUG_4,
                ENV_ID_PY_DEBUG_3,
                ENV_ID_PY_DEBUG_4,
            };
            Eigen::ArrayXd r_min = Eigen::ArrayXd(2);
            if (four_d_envs.contains(env_id)) {
                r_min = Eigen::ArrayXd(4);
            }
            r_min[0] = -1.0; 
            r_min[1] = -1.0; 
            if (four_d_envs.contains(env_id)) {
                r_min[2] = 0.0;
                r_min[3] = 0.0;
            }
            return max_steps * r_min;
        }

        if (env_id == ENV_ID_TOY_TREE_DENSE
            || env_id == ENV_ID_TOY_TREE_SPARSE)
        {
            double max_steps = 10.0;
            Eigen::ArrayXd r_min = Eigen::ArrayXd::Zero(2);
            return max_steps * r_min;
        }

        if (env_id == ENV_ID_VAMPLEW_DST
            || env_id == ENV_ID_VAMPLEW_STOCH_DST
            || env_id == ENV_ID_VAMPLEW_DST_MO_GYM
            || env_id == ENV_ID_VAMPLEW_STOCH_DST_MO_GYM
            || env_id == ENV_ID_VAMPLEW_DST_10
            || env_id == ENV_ID_VAMPLEW_STOCH_DST_10
            || env_id == ENV_ID_VAMPLEW_DST_10_CPP
            || env_id == ENV_ID_VAMPLEW_STOCH_DST_10_CPP)
        {
            double max_steps = get_max_trial_length();
            Eigen::ArrayXd r_min = Eigen::ArrayXd::Zero(2);
            r_min[0] = 0.0; 
            r_min[1] = -1.0; 
            return max_steps * r_min;
        }

        if (VARIABLE_SIZED_VAMPLEW_DST_ENVS.contains(env_id))
        {
            double max_steps = get_max_trial_length();
            Eigen::ArrayXd r_min = Eigen::ArrayXd::Zero(2);
            r_min[0] = 0.0; 
            r_min[1] = -1.0; 
            return max_steps * r_min;
        }

        if (env_id == ENV_ID_IMPROVED_DST
            || env_id == ENV_ID_IMPROVED_STOCH_DST)
        {
            double max_steps = get_max_trial_length();
            Eigen::ArrayXd r_min = Eigen::ArrayXd::Zero(3);
            r_min[0] = 0.0; 
            r_min[1] = -1.0; 
            r_min[2] = -9.0; 
            return max_steps * r_min;
        }

        if (VARIABLE_SIZED_IMPROVED_DST_ENVS.contains(env_id))
        {
            double max_steps = get_max_trial_length();
            Eigen::ArrayXd r_min = Eigen::ArrayXd::Zero(3);
            r_min[0] = 0.0; 
            r_min[1] = -1.0; 
            r_min[2] = -9.0; 
            return max_steps * r_min;
        }

        if (env_id == ENV_ID_FRUIT_TREE_7
            || env_id == ENV_ID_FRUIT_TREE_STOCH_5
            || env_id == ENV_ID_FRUIT_TREE_STOCH_7)
        {
            return Eigen::ArrayXd::Zero(6);
        }

        if (env_id == ENV_ID_RESOURCE_GATHER
            || env_id == ENV_ID_RESOURCE_GATHER_TIMED
            || env_id == ENV_ID_RESOURCE_GATHER_CPP
            || env_id == ENV_ID_RESOURCE_GATHER_TIMED_CPP)
        {
            Eigen::ArrayXd min_val = Eigen::ArrayXd(3);
            if (env_id == ENV_ID_RESOURCE_GATHER_TIMED) {
                min_val = Eigen::ArrayXd(4);
            }
            min_val[0] = -1.0;
            min_val[1] = 0.0;
            min_val[2] = 0.0;
            if (env_id == ENV_ID_RESOURCE_GATHER_TIMED) {
                min_val[3] = -1.0 * get_max_trial_length();
            }
            return min_val;
        }
        
        if (env_id == ENV_ID_BREAKABLE_BOTTLES) {
            Eigen::ArrayXd min_val = Eigen::ArrayXd(3);
            min_val[0] = -1.0 * get_max_trial_length();
            min_val[1] = 0.0;
            min_val[2] = -1.0; // max 2 bottles at a time, one can break, garuntee to deliver 2 bottles in at most 2 trips, even if try to take two on first trip
            return min_val;
        }

        if (env_id == ENV_ID_FOUR_ROOM
            || env_id == ENV_ID_FOUR_ROOM_TIMED) 
        {
            Eigen::ArrayXd min_val = Eigen::ArrayXd(3);
            if (env_id == ENV_ID_FOUR_ROOM_TIMED) {
                min_val = Eigen::ArrayXd(4);
            }
            min_val[0] = 0.0;
            min_val[1] = 0.0;
            min_val[2] = 0.0;
            if (env_id == ENV_ID_FOUR_ROOM_TIMED) {
                min_val[3] = -1.0 * get_max_trial_length();
            }
            return min_val;
        }

        if (env_id == ENV_ID_MINECART_DETERMINISTIC)
        {
            Eigen::ArrayXd min_val = Eigen::ArrayXd(3);
            min_val[0] = 0.0;
            min_val[1] = 0.0;
            min_val[2] = -1.0 * get_max_trial_length();
            return min_val;
        }

        throw runtime_error("get_env_value_lower_bound not implemented for env_id = " + env_id);
    }

    /**
     * Helper to add params to a manager args object for MoThtsManager level params
     */
    void RunManager::_add_thts_manager_params_to_args(MoThtsManagerArgs& manager_args, shared_ptr<MoThtsEnv> env)
    {
        manager_args.num_threads = get_num_search_threads();
        manager_args.num_envs = std::max(get_num_search_threads(), get_num_eval_threads());
        manager_args.max_depth = get_max_trial_length();
        manager_args.mcts_mode = get_mcts_mode();
        manager_args.graph_search = get_graph_search();
        manager_args.first_visit = true;
        manager_args.reward_dim = env->get_reward_dim();
        // MoThtsManager will load correct zero heuristic function based on reward dim if not set
        if (get_mcts_mode()) {
            manager_args.mo_heuristic_fn = make_shared<MoRolloutHeuristicFn>();
            manager_args.heuristic_psuedo_trials = 1;
        }
        manager_args.use_vector_visit_counts = get_vector_visit_counts();
        manager_args.convex_hull_max_size = get_convex_hull_max_size();
        manager_args.convex_hull_tolerance = get_convex_hull_tolerance();
        manager_args.use_solved_labelling = get_use_solved_labelling();
        manager_args.solved_labelling_fail_confidence = get_solved_labelling_fail_confidence();
        manager_args.solved_labelling_tolerance = get_solved_labelling_tolerance();
    }
    
    /**
     * Returns and instance of MoThtsManager to use for this run
     * Creates the manager and sets algorithm level parameters
     * Then adds the experiment level parameters and returns
    */
    shared_ptr<MoThtsManager> RunManager::get_thts_manager(shared_ptr<MoThtsEnv> env)
    {
        string alg_id = get_alg_id();
        shared_ptr<MoThtsManager> thts_manager = nullptr;

        if (alg_id == ALG_ID_CHVI) {
            MoThtsManagerArgs manager_args(env);
            _add_thts_manager_params_to_args(manager_args,env);
            return make_shared<MoThtsManager>(manager_args);
        }

        if (alg_id == ALG_ID_CZT || alg_id == ALG_ID_CZT_DOUBLING) 
        {
            CztManagerArgs manager_args(env);
            manager_args.bias = get_bias();
            if (alg_id == ALG_ID_CZT_DOUBLING)
            {
                manager_args.use_doubling_N_term = true;
                manager_args.min_log2_N = get_min_log2_N();
            }
            _add_thts_manager_params_to_args(manager_args,env);
            return make_shared<CztManager>(manager_args);
        }

        else if (alg_id == ALG_ID_CH_UCT)
        {
            ChUctManagerArgs manager_args(env);
            manager_args.bias = get_bias();
            _add_thts_manager_params_to_args(manager_args,env);
            return make_shared<ChUctManager>(manager_args);
        }

        else if (alg_id == ALG_ID_CH_CZT || alg_id == ALG_ID_CH_CZT_DOUBLING) 
        {
            ChCztManagerArgs manager_args(env);
            manager_args.bias = get_bias();
            if (alg_id == ALG_ID_CH_CZT_DOUBLING)
            {
                manager_args.use_doubling_N_term = true;
                manager_args.min_log2_N = get_min_log2_N();
            }
            _add_thts_manager_params_to_args(manager_args,env);
            return make_shared<ChCztManager>(manager_args);
        }

        else if (alg_id == ALG_ID_CH_BTS) // || alg_id == ALG_ID_CH_DENTS)
        {
            ChBtsManagerArgs manager_args(env);
            manager_args.temp_schedule_ptr = make_shared<SqrtSchedule>(get_init_temp(), get_temp_decay_rate());
            manager_args.epsilon = get_epsilon();
            manager_args.default_q_value = get_default_q_value();
            
            // if (alg_id == ALG_ID_CH_DENTS)
            // {
            //     manager_args.entropy_coeff_schedule_ptr = make_shared<LinearSchedule>(get_init_entropy_coeff(), get_entropy_zero_at());
            // }

            _add_thts_manager_params_to_args(manager_args,env);

            return make_shared<ChBtsManager>(manager_args);
        }

        else if (alg_id == ALG_ID_CH_HVUCT)
        {
            ChHvUctManagerArgs manager_args(env);
            manager_args.bias = get_bias();
            manager_args.hv_reference_point = make_shared<Vec>(get_env_value_lower_bound());
            _add_thts_manager_params_to_args(manager_args,env);
            return make_shared<ChHvUctManager>(manager_args);
        }

        else if (alg_id == ALG_ID_CH_PARETO)
        {
            ChParetoUctManagerArgs manager_args(env);
            manager_args.bias = get_bias();
            _add_thts_manager_params_to_args(manager_args,env);
            return make_shared<ChParetoUctManager>(manager_args);
        }

        else if (alg_id == ALG_ID_CH_CHEBY) {
            ChChebyUctManagerArgs manager_args(env);
            manager_args.bias = get_bias();
            _add_thts_manager_params_to_args(manager_args,env);
            return make_shared<ChChebyUctManager>(manager_args);
        }

        else if (alg_id == ALG_ID_CH_STANDARD_CHEBY) {
            ChChebyUctManagerArgs manager_args(env);
            manager_args.bias = get_bias();
            manager_args.use_standard_cheby_scalarization = true;
            manager_args.standard_cheby_reference_point = make_shared<Vec>(get_env_value_lower_bound());
            _add_thts_manager_params_to_args(manager_args,env);
            return make_shared<ChChebyUctManager>(manager_args);
        }

        stringstream ss;
        ss << "Error in RunManager get_thts_manager for alg_id = " << alg_id;
        throw runtime_error(ss.str());
    }

    /**
     * Returns a root node to use for search given these params
    */
    shared_ptr<MoThtsDNode> RunManager::get_root_search_node(shared_ptr<MoThtsEnv> env, shared_ptr<MoThtsManager> manager)
    {
        string alg_id = get_alg_id();
        if (alg_id == ALG_ID_CZT || alg_id == ALG_ID_CZT_DOUBLING) {
            shared_ptr<CztManager> czt_manager = static_pointer_cast<CztManager>(manager);
            return make_shared<CztDNode>(czt_manager, env->get_initial_state_itfc(), 0, 0);
        }   
        if (alg_id == ALG_ID_CH_UCT) {
            shared_ptr<ChUctManager> chuct_manager = static_pointer_cast<ChUctManager>(manager);
            return make_shared<ChUctDNode>(chuct_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_CH_CZT || alg_id == ALG_ID_CH_CZT_DOUBLING) {
            shared_ptr<ChCztManager> chczt_manager = static_pointer_cast<ChCztManager>(manager);
            return make_shared<ChCztDNode>(chczt_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_CH_BTS) {
            shared_ptr<ChBtsManager> chbts_manager = static_pointer_cast<ChBtsManager>(manager);
            return make_shared<ChBtsDNode>(chbts_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_CH_HVUCT) {
            shared_ptr<ChHvUctManager> chhvuct_manager = static_pointer_cast<ChHvUctManager>(manager);
            return make_shared<ChHvUctDNode>(chhvuct_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_CH_PARETO) {
            shared_ptr<ChParetoUctManager> chpareto_manager = static_pointer_cast<ChParetoUctManager>(manager);
            return make_shared<ChParetoUctDNode>(chpareto_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_CH_CHEBY || alg_id == ALG_ID_CH_STANDARD_CHEBY) {
            shared_ptr<ChChebyUctManager> chcheby_manager = static_pointer_cast<ChChebyUctManager>(manager);
            return make_shared<ChChebyUctDNode>(chcheby_manager, env->get_initial_state_itfc(), 0, 0);
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
    string RunManager::get_params_string_helper() 
    {
        stringstream ss;
        const vector<string>& relevant_alg_param_ids = ALG_ID_TO_ALG_PARAM_IDS.at(get_alg_id());
        bool first_iter = true;
        for (const string& alg_param_id : relevant_alg_param_ids)
        {
            if (!first_iter)
            {
                ss << "/";
            }
            first_iter = false;
            ss << alg_param_id << "=" << get_config_value<double>(alg_config, alg_param_id);
        }
        return ss.str();
    }

    /**
     * Gets the results directory for this run (doesn't check/make)
    */
    string RunManager::get_eval_logs_dir() 
    {
        stringstream ss;
        ss << "mo_eval_logs/";
        if (xpr_dir_override.empty()) {
            ss << get_xpr_name() << "_" << xpr_timestamp;
        } else {
            ss << xpr_dir_override;
        }
        ss << "/" << get_env_id() 
            << "/" << "env_size=" << get_env_size()
            << "/" << get_alg_id() 
            << "/" << get_params_string_helper();
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
        std::filesystem::path filename = get_eval_log_filename();
        std::filesystem::path dir = filename.parent_path();

        if (!std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
        }

        // Open the file (will create it if it doesn't exist)
        ofstream file(filename, ios::out | ios::trunc);
        if (!file.is_open()) 
        {
            throw runtime_error("Failed to open file: " + filename.string());
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
            << XPR_PARAM_ID_VECTOR_VISIT_COUNTS << ","
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
            << get_vector_visit_counts() << ","
            << get_max_trial_length() << ","
            << xpr_is_runtime_bounded() << ","
            << get_termination_bound() << ","
            << get_repeated_runs_per_alg() << ","
            << get_num_search_threads() << ","
            << get_eval_delta() << ","
            << get_num_eval_rollouts() << ","
            << get_num_eval_threads() << endl << endl;
        
        // Alg level params
        string alg_id = get_alg_id();
        fs << "Alg params: " << endl << endl;
        
        // Alg level param ids
        stringstream values_ss;
        fs << "alg_id";
        values_ss << get_alg_id();
        for (string alg_param_id : ALG_ID_TO_ALG_PARAM_IDS.at(alg_id))
        {
            fs << "," << alg_param_id;
            values_ss << "," << get_config_value<double>(alg_config, alg_param_id);
        }
        fs << endl;
        fs << values_ss.str();
        fs << endl << endl;

        // Header for main body
        fs << "Evals: " << endl << endl;
        fs << "run_idx,"
            << "ctx_mean,"
            << "ctx_std_dev,"
            << "reweighted_ctx_mean,"
            << "reweighted_ctx_std_dev,"
            << "normalised_ctx_mean,"
            << "normalised_ctx_std_dev,"
            << "hypervolume,"
            << "additive_eps_metric,"
            << "sparsity_metric,"
            << "normalised_hypervolume,"
            << "normalised_additive_eps_metric,"
            << "normalised_sparsity_metric,"
            << "num_trials,"
            << "num_backups,"
            << "runtime,"
            << "search_budget_consumed,"
            << "num_eval_samples" << endl;
    }

    void RunManager::write_eval_log_line(
        ofstream& fs, 
        int run_idx, 
        MoEvalMetrics& mo_eval_metrics, 
        int num_trials, 
        int num_backups,
        double runtime, 
        double search_budget_consumed, 
        int num_eval_samples)
    {
        fs << run_idx << "," 
            << mo_eval_metrics.ctx_mean << "," 
            << mo_eval_metrics.ctx_std_dev << "," 
            << mo_eval_metrics.reweighted_ctx_mean << "," 
            << mo_eval_metrics.reweighted_ctx_std_dev << "," 
            << mo_eval_metrics.normalised_ctx_mean << "," 
            << mo_eval_metrics.normalised_ctx_std_dev << "," 
            << mo_eval_metrics.hypervolume << "," 
            << mo_eval_metrics.additive_eps_metric << ","
            << mo_eval_metrics.sparsity_metric << ","
            << mo_eval_metrics.normalised_hypervolume << ","
            << mo_eval_metrics.normalised_additive_eps_metric << ","
            << mo_eval_metrics.normalised_sparsity_metric << ","
            << num_trials << "," 
            << num_backups << ","
            << runtime << "," 
            << search_budget_consumed << ","
            << num_eval_samples << endl;
    }


    std::filesystem::path RunManager::get_tree_log_filename(int run_idx)
    {

        stringstream filename_ss;
        filename_ss << "tree_log_run_" << _int_to_string_padded(run_idx, 4) << ".txt";

        std::filesystem::path dir = get_eval_logs_dir();
        std::filesystem::path filename = dir / filename_ss.str();

        return filename;
    }

    std::ofstream RunManager::get_tree_log_filestream(int run_idx)
    {

        std::filesystem::path filename = get_tree_log_filename(run_idx);
        std::filesystem::path dir = filename.parent_path();

        if (!std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
        }

        // Open the file (will create it if it doesn't exist)
        ofstream file(filename, ios::out | ios::trunc);
        if (!file.is_open()) 
        {
            throw runtime_error("Failed to open file: " + filename.string());
        }

        return file;
    }

    void RunManager::dump_tree_log(shared_ptr<MoThtsDNode> root_node, int run_idx)
    {
        ofstream tree_log_fs = get_tree_log_filestream(run_idx);
        tree_log_fs << root_node->get_pretty_print_string(2) << endl;
        tree_log_fs.close();
    }

    std::filesystem::path RunManager::get_convex_hull_log_filename(int run_idx)
    {

        stringstream filename_ss;
        filename_ss << "convex_hull_log_run_" << _int_to_string_padded(run_idx, 4) << ".txt";

        std::filesystem::path dir = get_eval_logs_dir();
        std::filesystem::path filename = dir / filename_ss.str();

        return filename;
    }

    std::ofstream RunManager::get_convex_hull_log_filestream(int run_idx)
    {

        std::filesystem::path filename = get_convex_hull_log_filename(run_idx);
        std::filesystem::path dir = filename.parent_path();

        if (!std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
        }

        // Open the file (will create it if it doesn't exist)
        ofstream file(filename, ios::out | ios::trunc);
        if (!file.is_open()) 
        {
            throw runtime_error("Failed to open file: " + filename.string());
        }

        return file;
    }

    void RunManager::dump_convex_hull_log(const ConvexHull& convex_hull, int run_idx)
    {
        ofstream convex_hull_log_fs = get_convex_hull_log_filestream(run_idx);
        convex_hull_log_fs << convex_hull << endl;
        convex_hull_log_fs.close();
    }
}
