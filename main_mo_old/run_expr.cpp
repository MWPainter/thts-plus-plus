#include "main_mo/run_expr.h"

#include "helper_templates.h"

#include "mo/mo_mc_eval.h"
#include "mo/mo_thts.h"
#include "py/mo_py_thts.h"
#include "py/py_multiprocessing_thts_env.h"

#include "mo/algorithms/contextual_zooming/czt_chance_node.h"
#include "mo/algorithms/contextual_zooming/czt_decision_node.h"
#include "mo/algorithms/chmcts/ch_czt_chance_node.h"
#include "mo/algorithms/chmcts/ch_czt_decision_node.h"
#include "mo/algorithms/simplex_maps/sm_chance_node.h"
#include "mo/algorithms/simplex_maps/sm_decision_node.h"

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

static const string HP_OPT_RESULTS_DIR = "hp_opt_mo/";

namespace thts {

    /**
     * Returns the results directory to use for this run, making sure that it exists, and creating it if it doesnt
    */
    string create_results_dir(RunID& run_id) {
        string results_dir = get_results_dir(run_id);
        if (!filesystem::exists(results_dir)) {
            filesystem::create_directories(results_dir);
        }
        return results_dir;
    }

    /**
     * Returns the filename for the mc eval results file
    */ 
    string get_mc_eval_results_filename(RunID& run_id) {
        stringstream ss;
        ss << get_results_dir(run_id)
            << "eval"
            << ".txt";
        return ss.str();
    }

    /**
     * Int to string with prepended zeros
    */
    string int_to_string_padded(int num, int pad_size=3) {
        stringstream ss;
        ss << setfill('0') << setw(pad_size) << num;
        return ss.str();
    }

    /**
     * Returns the filename for a tree printout
    */
    string get_tree_filename(RunID& run_id, int replicate) {
        stringstream ss;
        ss << get_results_dir(run_id)
            << "tree_"
            << replicate
            << ".txt";
        return ss.str();
    }

    /**
     * Returns the filename for a convex hull printout
    */
    string get_convex_hull_filename(RunID& run_id, int replicate) {
        stringstream ss;
        ss << get_results_dir(run_id)
            << "convex_hull_"
            << replicate
            << ".txt";
        return ss.str();
    }

    /**
     * Writes the param header to a file
    */
    void write_param_header_to_file(RunID& run_id, ofstream& out_file) {
        stringstream ss_names;
        stringstream ss_values;
        for (pair<string,double> param_val_entry : run_id.alg_params) {
            ss_names << param_val_entry.first << ",";
            ss_values << param_val_entry.second << ",";
        }

        ss_names << "alg";
        ss_values << run_id.alg_id;

        out_file << ss_names.str() << endl
            << ss_values.str() << endl
            << endl;
    }

    /**
     * Write the eval header to the eval file
    */
    void write_eval_header(ofstream& eval_out_file) {
        eval_out_file << "replicate,"
            << "search_time" << ","
            << "num_trials" << ","
            << "eum_mean" << ","
            << "eum_var" << ","
            << "norm_eum_mean" << ","
            << "norm_eum_var" << ","
            << "rand_eum_mean" << ","
            << "rand_eum_var" << ","
            << "rand_norm_eum_mean" << ","
            << "rand_norm_eum_var" << ","
            << "hypervolume_metric" << "," 
            << "sparsity_metric" << "," 
            << "additive_eps_metric" << endl;
    }

    /**
     * Write the results from an evaluation to the eval file
    */
    void write_eval_line(
        ofstream& eval_out_file, 
        int replicate, 
        double search_time, 
        int num_trials, 
        double eum_mean, 
        double eum_var, 
        double norm_eum_mean, 
        double norm_eum_var,
        double rand_eum_mean,
        double rand_eum_var,
        double rand_norm_eum_mean,
        double rand_norm_eum_var,
        double hypervolume_metric,
        double sparsity_metric,
        double additive_eps_metric)
    {
        eval_out_file 
            << replicate << ","
            << search_time << ","
            << num_trials << ","
            << eum_mean << ","
            << eum_var << ","
            << norm_eum_mean << ","
            << norm_eum_var << ","
            << rand_eum_mean << ","
            << rand_eum_var << ","
            << rand_norm_eum_mean << ","
            << rand_norm_eum_var << ","
            << hypervolume_metric << "," 
            << sparsity_metric << "," 
            << additive_eps_metric << endl;
    }

    /**
     * Perform an mc eval (with either randomly sampled or well spaced points from simplex)
     * Returns the mean and std via the double& values
    */
    void run_mc_eval(
        double& mean, 
        double& var, 
        double& normalised_mean, 
        double& normalised_var, 
        shared_ptr<MoThtsEnv> env, 
        shared_ptr<MoThtsDNode> root_node, 
        shared_ptr<MoThtsManager> thts_manager,
        RunID& run_id,
        bool well_spaced) 
    {   
        shared_ptr<EvalPolicy> eval_policy = make_shared<EvalPolicy>(root_node, env, thts_manager);
        MoMCEvaluator evaluator(
            eval_policy, 
            run_id.max_trial_length, 
            thts_manager, 
            run_id.get_env_min_value(), 
            run_id.get_env_max_value(),
            well_spaced);
        evaluator.run_rollouts(run_id.rollouts_per_mc_eval, run_id.eval_threads);
        mean = evaluator.get_mo_ctx_return_mean();
        var = evaluator.get_mo_ctx_return_variance();
        normalised_mean = evaluator.get_normalised_mo_ctx_return_mean();
        normalised_var = evaluator.get_normalised_mo_ctx_return_variance();
    }

    /**
     * Compute evals
     */
    void compute_evals(
        double& eum_mean,
        double& eum_var,
        double& norm_eum_mean,
        double& norm_eum_var,
        double& rand_eum_mean,
        double& rand_eum_var,
        double& rand_norm_eum_mean,
        double& rand_norm_eum_var,
        double& hypervolume_metric,
        double& sparsity_metric,
        double& additive_eps_metric,
        shared_ptr<MoThtsEnv> env, 
        shared_ptr<MoThtsDNode> root_node, 
        shared_ptr<MoThtsManager> thts_manager,
        RunID& run_id,
        bool just_well_spaced) 
    {
        // MC eval with well spaced context weights
        run_mc_eval(
            eum_mean, 
            eum_var, 
            norm_eum_mean, 
            norm_eum_var, 
            env, 
            root_node, 
            thts_manager, 
            run_id, 
            true);
        
        if (just_well_spaced) {
            return;
        }

        // MC eval with randomly sampled context weights
        run_mc_eval(
            rand_eum_mean, 
            rand_eum_var, 
            rand_norm_eum_mean, 
            rand_norm_eum_var, 
            env, 
            root_node, 
            thts_manager, 
            run_id, 
            false);

        // Compute convex hull metrics 
        ConvexHull root_node_ch = root_node->get_convex_hull();
        hypervolume_metric = root_node_ch.hypervolume(run_id.get_env_min_value());
        sparsity_metric = root_node_ch.sparsity_metric();
        additive_eps_metric = root_node_ch.additive_eps_metric();
    }

    /**
     * Returns the filename for a debug info printout
    */
    string get_debug_filename(RunID& run_id, int replicate) {
        stringstream ss;
        ss << get_results_dir(run_id)
            << "debug_info"
            // << "tree_"
            // << get_params_string_helper(run_id)
            << ".txt";
        return ss.str();
    }

    /**
     * Writes debug info
    */
    void write_debug_info_to_file(shared_ptr<MoThtsDNode> root_node, ofstream& out_file) {
        shared_ptr<CztDNode> ball_list_root_node = dynamic_pointer_cast<CztDNode>(root_node);
        if (ball_list_root_node) {
            out_file << "ROOT NODE INFO" << endl   
                << "---------------" << endl;
            out_file << ball_list_root_node->get_ball_list_pretty_print_string() << endl;
            out_file << "---------------" << endl;

            for (pair<const shared_ptr<const Action>,shared_ptr<ThtsCNode>>& child_pair : root_node->children) {
                out_file << "ACTION " << child_pair.first << endl 
                    << "---------------" << endl;
                CztCNode& child_node = (CztCNode&) *child_pair.second;
                out_file << child_node.get_ball_list_pretty_print_string() << endl;
                out_file << "---------------" << endl;
            }  
        }

        shared_ptr<ChCztDNode> convex_hull_root_node = dynamic_pointer_cast<ChCztDNode>(root_node);
        if (convex_hull_root_node) {
            out_file << "ROOT NODE INFO" << endl   
                << "---------------" << endl;
            out_file << convex_hull_root_node->get_convex_hull_pretty_print_string() << endl;
            out_file << "---------------" << endl;

            for (pair<const shared_ptr<const Action>,shared_ptr<ThtsCNode>>& child_pair : root_node->children) {
                out_file << "ACTION " << child_pair.first << endl 
                    << "---------------" << endl;
                ChCztCNode& child_node = (ChCztCNode&) *child_pair.second;
                out_file << child_node.get_convex_hull_pretty_print_string() << endl;
                out_file << "---------------" << endl;
            }  
        }

        shared_ptr<SmThtsDNode> simplex_map_root_node = dynamic_pointer_cast<SmThtsDNode>(root_node);
        if (simplex_map_root_node) {
            out_file << "ROOT NODE INFO" << endl   
                << "---------------" << endl;
            out_file << simplex_map_root_node->get_simplex_map_pretty_print_string() << endl;
            out_file << "---------------" << endl;

            for (pair<const shared_ptr<const Action>,shared_ptr<ThtsCNode>>& child_pair : root_node->children) {
                out_file << "ACTION " << child_pair.first << endl 
                    << "---------------" << endl;
                SmThtsCNode& child_node = (SmThtsCNode&) *child_pair.second;
                out_file << child_node.get_simplex_map_pretty_print_string() << endl;
                out_file << "---------------" << endl;
            }  
        }
    }

    /**
     * Performs all of the (replicated) runs corresponding to 'run_id'
     * (This is the one exposed function (for now) in run_toy.cpp)
     * 
     * We dont try to gracefully exit
    */
    vector<double> run_expr(RunID& run_id, bool hp_opt, int hp_opt_replicate) {
        // create results dir + eval file
        if (!hp_opt) {
            create_results_dir(run_id);
        }
        string eval_filename = get_mc_eval_results_filename(run_id);
        ofstream eval_file;
        if (!hp_opt) {
            eval_file.open(eval_filename, ios::out);// | ios::app);
            write_param_header_to_file(run_id, eval_file);
            write_eval_header(eval_file);
        }

        // Run experiment 'replicate' many times
        vector<double> mc_eval_mean_and_vars = vector<double>(run_id.num_repeats * 2);
        for (int replicate=0; replicate<run_id.num_repeats; replicate++) {

            // print
            cout << "Starting run on " << run_id.env_id << " with alg " << run_id.alg_id << " and params " 
                << helper::unordered_map_pretty_print_string(run_id.alg_params) << ", replicate ";
            if (!hp_opt) {
                cout << replicate << endl;
            } else {
                cout << hp_opt_replicate << endl;
            }
                
            // setup env
            shared_ptr<MoThtsEnv> env = run_id.get_env();
            shared_ptr<MoThtsManager> thts_manager = run_id.get_thts_manager(env);
            if (run_id.is_python_env()) {
                for (int i=0; i<run_id.num_envs; i++) {
                    PyMultiprocessingThtsEnv& py_mp_env = *dynamic_pointer_cast<PyMultiprocessingThtsEnv>(
                        thts_manager->thts_env(i));
                    py_mp_env.start_python_server(i);
                }
            }
            shared_ptr<MoThtsDNode> root_node = run_id.get_root_search_node(env, thts_manager);
            shared_ptr<ThtsPool> thts_pool = make_shared<MoThtsPool>(thts_manager, root_node, run_id.num_threads);

            // eval at 0 trials
            double eum, eum_var, norm_eum, norm_eum_var;
            double rand_eum, rand_eum_var, rand_norm_eum, rand_norm_eum_var;
            double hypervolume_metric, sparsity_metric, additive_eps_metric;
            if (!hp_opt) {
                compute_evals(
                    eum, 
                    eum_var, 
                    norm_eum, 
                    norm_eum_var, 
                    rand_eum, 
                    rand_eum_var, 
                    rand_norm_eum, 
                    rand_norm_eum_var, 
                    hypervolume_metric,
                    sparsity_metric,
                    additive_eps_metric,
                    env, 
                    root_node, 
                    thts_manager, 
                    run_id,
                    false);
                write_eval_line(
                    eval_file, 
                    replicate, 
                    0.0, 
                    0, 
                    eum, 
                    eum_var, 
                    norm_eum, 
                    norm_eum_var, 
                    rand_eum, 
                    rand_eum_var, 
                    rand_norm_eum, 
                    rand_norm_eum_var, 
                    hypervolume_metric,
                    sparsity_metric,
                    additive_eps_metric);
            }

            // run trials, evaluating every eval delta
            double search_time_elapsed = 0.0;
            while (search_time_elapsed < run_id.search_runtime) {
                thts_pool->run_trials(numeric_limits<int>::max(), run_id.eval_delta);
                search_time_elapsed += run_id.eval_delta;
                compute_evals(
                    eum, 
                    eum_var, 
                    norm_eum, 
                    norm_eum_var, 
                    rand_eum, 
                    rand_eum_var, 
                    rand_norm_eum, 
                    rand_norm_eum_var, 
                    hypervolume_metric,
                    sparsity_metric,
                    additive_eps_metric,
                    env, 
                    root_node, 
                    thts_manager, 
                    run_id,
                    hp_opt);
                if (!hp_opt) {
                    write_eval_line(
                        eval_file, 
                        replicate, 
                        search_time_elapsed, 
                        root_node->get_scalar_num_visits(), 
                        eum, 
                        eum_var, 
                        norm_eum, 
                        norm_eum_var, 
                        rand_eum, 
                        rand_eum_var, 
                        rand_norm_eum, 
                        rand_norm_eum_var, 
                        hypervolume_metric,
                        sparsity_metric,
                        additive_eps_metric);
                }
            }

            if (!hp_opt) {
                // Write tree to file
                string tree_filename = get_tree_filename(run_id, replicate);
                ofstream tree_file;
                tree_file.open(tree_filename, ios::out);
                tree_file << root_node->get_pretty_print_string(1) << endl;
                tree_file.close();

                // Write debug info
                string debug_filename = get_debug_filename(run_id, replicate);
                ofstream debug_file;
                debug_file.open(debug_filename, ios::out);
                write_debug_info_to_file(root_node, debug_file);
                debug_file.close();

                // Write convex hull
                string convex_hull_filename = get_convex_hull_filename(run_id, replicate);
                ofstream ch_file;
                ch_file.open(convex_hull_filename, ios::out);
                ch_file << root_node->get_convex_hull() << endl;
                ch_file.close();

                // Flush
                eval_file.flush();
            }

            // Update results
            mc_eval_mean_and_vars[replicate*2] = eum;
            mc_eval_mean_and_vars[replicate*2+1] = eum_var;
            
            env.reset();
            thts_manager.reset();
            root_node.reset();
            thts_pool.reset();
        }   

        // close eval file
        if (!hp_opt) {
            eval_file.close();
        }

        // Return avg mean utility over replicates
        return mc_eval_mean_and_vars;
    }

    /**
     * Runs all experiments in 'run_ids'
     * This was in main, until needed to start hacking in the sill subinterpreter bug mitigation
     * 
     * (Entry point for 'eval' experiments)
    */
    void run_exprs(shared_ptr<vector<RunID>> run_ids) 
    {
        // Check if any run ids need python
        bool need_python = false;
        for (RunID& run_id : *run_ids) {
            if (run_id.is_python_env()) {
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
        for (RunID& run_id : *run_ids) {
            thts::run_expr(run_id);
        }
    }

    /**
     * Returns the results directory to use for this run, making sure that it exists, and creating it if it doesnt
    */
    void create_hp_opt_results_dir() {
        if (!filesystem::exists(HP_OPT_RESULTS_DIR)) {
            filesystem::create_directories(HP_OPT_RESULTS_DIR);
        }
    }

    /**
     * Returns the filename for the mc eval results file (summary of hp opt)
    */ 
    string get_hp_opt_summary_filename(string expr_id, time_t timestamp) {
        stringstream ss;
        ss << HP_OPT_RESULTS_DIR << expr_id << "_summary_" << timestamp << ".txt";
        return ss.str();
    }

    /**
     * Returns the filename for the mc eval results file (all evaluations of each params)
    */ 
    string get_hp_opt_evals_filename(string expr_id, time_t timestamp) {
        stringstream ss;
        ss << HP_OPT_RESULTS_DIR << expr_id << "_evals_" << timestamp << ".txt";
        return ss.str();
    }


    /**
     * Runs hyperparameter opt for 'expr_id'
     */
    void run_hp_opt(string expr_id_prefix) {
        // Lookup expr_id
        string expr_id = lookup_expr_id_from_prefix(expr_id_prefix);

        // timestamp, so can rerun with same params and keep both results
        time_t expr_timestamp = std::time(nullptr);
        
        // Create output filestream
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