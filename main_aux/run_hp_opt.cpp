#include "main_aux/run_xpr.h"

#include "helper_templates.h"

#include "mc_eval.h"

#include "thts.h"
#include "py/py_thts.h"
#include "py/py_multiprocessing_thts_env.h"

#include "py/py_helper.h"
#include <Python.h>

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
     * Runs hyperparameter opt for 'expr_id'
     */
    void run_hp_opt(string expr_id_prefix) {
        // Lookup expr_id
        string expr_id = lookup_expr_id_from_prefix(expr_id_prefix);

        // timestamp, so can rerun with same params and keep both results
        time_t expr_timestamp = std::time(nullptr);
        
        // Create output filestreams
        create_hp_opt_results_dir();

        string hp_opt_summary_filename = get_hp_opt_summary_filename(expr_id, expr_timestamp);
        ofstream hp_opt_summary_file;
        hp_opt_summary_file.open(hp_opt_summary_filename, ios::out);// | ios::app);

        string hp_opt_evals_filename = get_hp_opt_evals_filename(expr_id, expr_timestamp);
        ofstream hp_opt_evals_file;
        hp_opt_evals_file.open(hp_opt_evals_filename, ios::out);// | ios::app);

        // Get the hp_opt
        shared_ptr<HyperparamOptimiser> hp_opt = get_hyperparam_optimiser_from_expr_id(
            expr_id, expr_timestamp, hp_opt_summary_file, hp_opt_evals_file);

        // If running python, make interpreter and release gil
        shared_ptr<py::scoped_interpreter> py_interpreter;
        shared_ptr<py::gil_scoped_release> release;
        if (hp_opt->is_python_env()) {
            py_interpreter = make_shared<py::scoped_interpreter>();
            release = make_shared<py::gil_scoped_release>();
        }

        // Write header
        hp_opt->write_header();

        // Run bayesopt (we do own logging, so results vector unecessary)
        bayesopt::vectord _results(hp_opt->num_hyperparams);
        hp_opt->optimize(_results);

        // Write best eval to file at end
        hp_opt->write_best_eval();

        // Close files
        hp_opt_summary_file.close();
        hp_opt_evals_file.close();
    }
}





// #include "main_aux/run_xpr.h"

// #include "helper_templates.h"

// #include "mc_eval.h"

// #include "thts.h"
// #include "py/py_thts.h"
// #include "py/py_multiprocessing_thts_env.h"

// #include "py/py_helper.h"
// #include <Python.h>

// #include <filesystem>
// #include <fstream>
// #include <iomanip>
// #include <iostream>
// #include <sstream>
// #include <string>

// #include <pybind11/pybind11.h>
// #include <pybind11/embed.h>

// using namespace std;
// namespace py = pybind11;
// using namespace thts;
// using namespace thts::python;

// namespace thts {

//     /**
//      * Entry point, xpr_id_prefix from command line
//      * Checks if any run's need python
//      * If so, makes an interpreter and releases gil
//      */
//     void main_xpr(string xpr_id_prefix)
//     {
//         // Read in config
//         vector<ConfigMap> xpr_configs = RunManager::lookup_config_vector_from_xpr_prefix(xpr_id_prefix);
//         vector<RunManager> run_managers = RunManager::get_run_managers_from_config_vector(xpr_configs);

//         // Check if any run ids need python
//         bool need_python = false;
//         for (RunManager& run_manager : run_managers) {
//             if (run_manager.is_python_env()) {
//                 need_python = true;
//                 break;
//             }
//         }

//         // If running python, make interpreter and release gil
//         shared_ptr<py::scoped_interpreter> py_interpreter;
//         shared_ptr<py::gil_scoped_release> release;
//         if (need_python) {
//             py_interpreter = make_shared<py::scoped_interpreter>();
//             release = make_shared<py::gil_scoped_release>();
//         }   

//         // Actually run experiments
//         for (RunManager& run_manager : run_managers) {
//             _run_expr(run_id);
//         }
//     }

//     /**
//      * Performs all of the (replicated) runs corresponding to 'run_id'
//     */
//     vector<double> _run_expr(RunManager& run_manager, int run_idx, bool log_evals=true, bool log_trees=true)
//     {
//         // Open eval log
//         ofstream eval_log_fs;
//         if (log_evals)
//         {
//             eval_log_fs = run_manager.get_eval_log_filestream();
//             run_manager.write_eval_log_header(eval_log_fs);
//         }
        

//         // Run experiment 'replicate' many times
//         vector<double> value_estimates = vector<double>(run_id.num_repeats);
//         for (int replicate=0; replicate<run_id.num_repeats; replicate++) {

//             // print
//             cout << "Starting run on " << run_id.env_id << " with alg " << run_id.alg_id << " and params " 
//                 << helper::unordered_map_pretty_print_string(run_id.alg_params) << ", replicate ";
//             if (!hp_opt) {
//                 cout << replicate << endl;
//             } else {
//                 cout << hp_opt_replicate << endl;
//             }
                
//             // setup env
//             shared_ptr<ThtsEnv> env = run_id.get_env();
//             shared_ptr<ThtsManager> thts_manager = run_id.get_thts_manager(env);
//             if (run_id.is_python_env()) {
//                 for (int i=0; i<run_id.num_envs; i++) {
//                     PyMultiprocessingThtsEnv& py_mp_env = *dynamic_pointer_cast<PyMultiprocessingThtsEnv>(
//                         thts_manager->thts_env(i));
//                     py_mp_env.start_python_server(i);
//                 }
//             }
//             shared_ptr<ThtsDNode> root_node = run_id.get_root_search_node(env, thts_manager);
//             shared_ptr<ThtsPool> thts_pool = make_shared<ThtsPool>(thts_manager, root_node, run_id.num_threads);

//             // eval at 0 trials
//             double mean, stddev;
//             if (!hp_opt) {
//                 run_mc_eval(
//                     mean, 
//                     stddev, 
//                     env, 
//                     root_node, 
//                     thts_manager, 
//                     run_id);
//                 write_eval_line(eval_file, replicate, 0.0, 0, mean, stddev);
//             }

//             // run trials, evaluating every eval delta
//             double search_time_elapsed = 0.0;
//             while (search_time_elapsed < run_id.search_runtime) {
//                 int max_trials = numeric_limits<int>::max();
//                 double max_runtime = numeric_limits<double>::max();
//                 if (run_id.eval_wrt_time) {
//                     max_runtime = run_id.eval_delta;
//                 } else {
//                     max_trials = run_id.eval_delta;
//                 }
//                 thts_pool->run_trials(max_trials, max_runtime);
//                 search_time_elapsed += run_id.eval_delta;
//                 run_mc_eval(
//                     mean, 
//                     stddev, 
//                     env, 
//                     root_node, 
//                     thts_manager, 
//                     run_id);
//                 if (!hp_opt) {
//                     write_eval_line(
//                         eval_file, 
//                         replicate, 
//                         search_time_elapsed, 
//                         root_node->get_num_visits(), 
//                         mean, 
//                         stddev);
//                 }
//             }

//             if (!hp_opt) {
//                 // Write tree to file
//                 if (replicate == 0) {
//                     string tree_filename = get_tree_filename(run_id, replicate);
//                     ofstream tree_file;
//                     tree_file.open(tree_filename, ios::out);
//                     tree_file << root_node->get_pretty_print_string(1) << endl;
//                     tree_file.close();
//                 }

//                 // Write debug info
//                 if (replicate == 0) {
//                     string debug_filename = get_debug_filename(run_id, replicate);
//                     ofstream debug_file;
//                     debug_file.open(debug_filename, ios::out);
//                     write_debug_info_to_file(root_node, debug_file);
//                     debug_file.close();
//                 }

//                 // Flush
//                 eval_file.flush();
//             }

//             // Update results
//             value_estimates[replicate] = mean;
            
//             env.reset();
//             thts_manager.reset();
//             root_node.reset();
//             thts_pool.reset();
//         }   

//         // close eval file
//         if (!hp_opt) {
//             eval_file.close();
//         }

//         // Return avg mean utility over replicates
//         return value_estimates;
//     }

//     /**
//      * Perform an mc eval (of policy from tree node)
//     */
//     pair<double,double> _mc_eval(
//         shared_ptr<ThtsEnv> env, 
//         shared_ptr<ThtsDNode> root_node, 
//         shared_ptr<ThtsManager> thts_manager,
//         RunManager& run_manager) 
//     {   
//         shared_ptr<EvalPolicy> eval_policy = make_shared<EvalPolicy>(root_node, env, thts_manager);
//         MCEvaluator evaluator(eval_policy, run_manager.get_max_trial_length(), thts_manager);
//         evaluator.run_rollouts(run_manager.get_num_eval_rollouts(), run_manager.get_num_eval_threads());
//         double mean = evaluator.get_mean_return();
//         double std_dev = evaluator.get_stddev_return();
//         return make_pair(mean,std_dev);
//     }

// }