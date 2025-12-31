#include "main_aux/run_xpr.h"

#include "helper_templates.h"
#include "mo/mo_helper_templates.h"

#include "mo/mo_mc_eval.h"

#include "mo/mo_thts.h"
#include "py/mo_py_multiprocessing_thts_env.h"

#include "py/py_helper.h"
#include <Python.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

using namespace std;
namespace py = pybind11;
using namespace thts;
using namespace thts::python;

namespace thts {

    /**
     * Entry point, xpr_id_prefix from command line
     * Checks if any run's need python
     * If so, makes an interpreter and releases gil
     */
    void main_xpr(string xpr_id_prefix, string xpr_dir_override)
    {
        // Read in config
        vector<ConfigMap> xpr_configs = RunManager::lookup_config_vector_from_xpr_prefix(xpr_id_prefix);
        shared_ptr<vector<RunManager>> run_managers_ptr = RunManager::get_run_managers_from_config_vector(
            xpr_configs, xpr_dir_override);
        vector<RunManager>& run_managers = *run_managers_ptr;

        // Check if any run ids need python
        bool need_python = false;
        for (RunManager& run_manager : run_managers) {
            if (run_manager.is_python_env()) {
                need_python = true;
                break;
            }
        }

        // If running python, make interpreter and release gil
        shared_ptr<py::scoped_interpreter> py_interpreter;
        shared_ptr<py::gil_scoped_release> release;
        if (need_python) {
            py_interpreter = make_shared<py::scoped_interpreter>();
            release = make_shared<py::gil_scoped_release>();
        }   

        // Actually run experiments
        for (RunManager& run_manager : run_managers) {
            run_searches(run_manager);
        }
    }

    /**
     * Performs all of the (replicated) searches corresponding to 'run_id'
     * If hpopt is true, then dont run any logging, and only return the final mc eval
    */
    double run_searches(RunManager& run_manager, bool hpopt, bool log_trees)
    {
        // Open eval log
        ofstream eval_log_fs;
        if (!hpopt)
        {
            eval_log_fs = run_manager.get_eval_log_filestream();
            run_manager.write_eval_log_header(eval_log_fs);
        }

        // final eval to return
        double final_eval_mean = 0.0; 
        
        // Run the perscribed number of repeats
        for (int run_idx=0; run_idx < run_manager.get_repeated_runs_per_alg(); run_idx++)
        {
            // cout so know we're doing something
            if (!hpopt)
            {
                cout << "Starting run on " << run_manager.get_env_id() << " with alg " << run_manager.get_alg_id() << " and params " 
                    << run_manager.get_params_string_helper() << ", run_idx = " << run_idx << endl;
            }

            // Variables for "runtime"
            int total_trials_run = 0;
            double total_runtime = 0.0;
            double search_budget_consumed = 0.0;

            // get env and manager
            shared_ptr<MoThtsEnv> env = run_manager.get_env();
            shared_ptr<MoThtsManager> thts_manager = run_manager.get_thts_manager(env);

            // (If python env) start up multiprocessing servers
            if (run_manager.is_python_env())
            {
                int num_search_threads = run_manager.get_num_search_threads();
                int num_eval_threads = run_manager.get_num_eval_threads();
                int num_envs_required = std::max(num_eval_threads, num_search_threads);

                for (int i=0; i < num_envs_required; i++) 
                {
                    MoPyMultiprocessingThtsEnv& py_mp_env = *dynamic_pointer_cast<MoPyMultiprocessingThtsEnv>(
                        thts_manager->thts_env(i));
                    py_mp_env.start_python_server(i);
                }
            }

            // Setup search
            shared_ptr<MoThtsDNode> root_node = run_manager.get_root_search_node(env, thts_manager);
            shared_ptr<MoThtsPool> thts_pool = make_shared<MoThtsPool>(thts_manager, root_node, run_manager.get_num_search_threads());

            // Eval at 0 trials
            double eval_mean = 0.0, eval_std = 0.0;
            if (!hpopt)
            {
                pair<double,double> eval = mc_eval(env, root_node, thts_manager, run_manager);
                eval_mean = eval.first;
                eval_std = eval.second;
                run_manager.write_eval_log_line(
                    eval_log_fs, run_idx, eval_mean, eval_std, 0.0, 0.0, 0.0, run_manager.get_num_eval_rollouts());
            }

            // run trials, evaluating every eval delta
            while (search_budget_consumed < run_manager.get_termination_bound())
            {
                // get budget to consume now
                int max_trials = numeric_limits<int>::max();
                double max_runtime = numeric_limits<double>::max();
                if (run_manager.xpr_is_runtime_bounded()) {
                    max_runtime = run_manager.get_eval_delta();
                } else {
                    max_trials = run_manager.get_eval_delta();
                }

                // run some trials
                auto start_timestamp = std::chrono::steady_clock::now();
                thts_pool->run_trials(max_trials, max_runtime);
                auto end_timestamp = std::chrono::steady_clock::now();

                // Update runtimes
                total_trials_run = thts_pool->get_total_trials_run();
                total_runtime += std::chrono::duration<double>(end_timestamp - start_timestamp).count();
                search_budget_consumed += run_manager.get_eval_delta();

                // eval (always run final eval, but only log if 'run_evals')
                if (!hpopt || search_budget_consumed >= run_manager.get_termination_bound())
                {
                    // TODO: update to use all MO eval metrics + log them
                    pair<double,double> eval = mc_eval(env, root_node, thts_manager, run_manager);
                    eval_mean = eval.first;
                    final_eval_mean = eval_mean;
                    eval_std = eval.second;
                    if (!hpopt)
                    {
                        run_manager.write_eval_log_line(
                            eval_log_fs, 
                            run_idx, 
                            eval_mean, 
                            eval_std, 
                            total_trials_run, 
                            total_runtime, 
                            search_budget_consumed, 
                            run_manager.get_num_eval_rollouts());
                    } 
                }
            }

            // Log debug info if wanted
            if (log_trees)
            {
                run_manager.dump_tree_log(root_node, run_idx);
            }

            // Flush
            if (!hpopt)
            {
                eval_log_fs.flush();
            }
            
            // Release resources in reverse order
            // (iirc, not doing this can cause python resources to be released without holding gil and segfaults)
            env.reset();
            thts_manager.reset();
            root_node.reset();
            thts_pool.reset();
        }   

        // close eval file
        if (!hpopt) 
        {
            eval_log_fs.close();
        }
        
        return final_eval_mean;
    }

    /**
     * Perform an mc eval (of policy from tree node)
    */
    pair<double,double> mc_eval(
        shared_ptr<MoThtsEnv> env, 
        shared_ptr<MoThtsDNode> root_node, 
        shared_ptr<MoThtsManager> thts_manager,
        RunManager& run_manager) 
    {   
        // TODO: update this to include r_min and r_max in MoMCEvaluator
        // TODO: update this to return the MO eval metrics
        shared_ptr<EvalPolicy> eval_policy = make_shared<EvalPolicy>(root_node, env, thts_manager);
        MoMCEvaluator evaluator(eval_policy, run_manager.get_max_trial_length(), thts_manager);
        evaluator.run_rollouts(run_manager.get_num_eval_rollouts(), run_manager.get_num_eval_threads());
        double mean = evaluator.get_mean_return();
        double std_dev = evaluator.get_stddev_return();
        return make_pair(mean,std_dev);
    }

    // TODO: add functions for other eval metrics - e.g. hypervolume

}