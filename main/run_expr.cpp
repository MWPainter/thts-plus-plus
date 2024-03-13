#include "main/run_expr.h"

#include "helper_templates.h"
#include "mo/mo_mc_eval.h"
#include "py/mo_py_mc_eval.h"
#include "mo/mo_thts.h"
#include "py/mo_py_thts.h"
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
     * Gets the results directory for this run (doesn't check/make)
    */
    string get_results_dir(RunID& run_id) {
        stringstream ss;
        ss << "results/" 
            << run_id.env_id << "/" 
            << run_id.expr_id << "_" << run_id.expr_timestamp << "/" 
            << run_id.alg_id << "/";
        return ss.str();
    }

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
     * Helper to make a string of:
     * "param1=val1,param2=val2,...,paramN=valN"
    */
    string get_params_string_helper(RunID& run_id) {
        stringstream ss;
        unsigned int i = 0;
        for (pair<string,double> param_val_entry : run_id.alg_params) {
            ss << param_val_entry.first << "=" << param_val_entry.second;
            if (++i < run_id.alg_params.size()) {
                ss << ",";
            }
        }
        return ss.str();
    }

    /**
     * Returns the filename for the mc eval results file
    */
    string get_mc_eval_results_filename(RunID& run_id) {
        stringstream ss;
        ss << get_results_dir(run_id)
            << "eval"
            // << "_"
            // << get_params_string_helper(run_id)
            << ".csv";
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
     * Returns the filename for the logger results file
    */
    string get_logger_results_filename(RunID& run_id, int replicate) {
        stringstream ss;
        ss << get_results_dir(run_id)
            << "log_"
            << get_params_string_helper(run_id) << "_"
            << int_to_string_padded(replicate)
            << ".csv";
        return ss.str();
    }

    /**
     * Returns the filename for a tree printout
    */
    string get_tree_filename(RunID& run_id, int replicate) {
        stringstream ss;
        ss << get_results_dir(run_id)
            << "tree_"
            << get_params_string_helper(run_id)
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

        out_file << ss_names.str() << "\n"
            << ss_values.str() << "\n"
            << endl;
    }

    /**
     * Write the eval header to the eval file
    */
    void write_eval_header(ofstream& eval_out_file) {
        eval_out_file << "replicate,"
            << "search_time,"
            << "num_trials,"
            << "mc_eval_utility_mean,"
            << "mc_eval_utility_std,"
            << "mc_eval_normalised_utility_mean"
            << "mc_eval_normalised_utility_std" << endl;
    }

    /**
     * Write the results from an evaluation to the eval file
    */
    void write_eval_line(
        ofstream& eval_out_file, 
        int replicate, 
        double search_time, 
        int num_trials, 
        double mc_eval_utility_mean, 
        double mc_eval_utility_std, 
        double mc_eval_normalised_utility_mean, 
        double mc_eval_normalised_utility_std) 
    {
        eval_out_file 
            << replicate << ","
            << search_time << ","
            << num_trials << ","
            << mc_eval_utility_mean << ","
            << mc_eval_utility_std << ","
            << mc_eval_normalised_utility_mean << ","
            << mc_eval_normalised_utility_std << endl;
    }

    /**
     * Perform an mc eval
     * Returns the mean and std via the double& values
    */
    void run_mc_eval(
        double& mean, 
        double& std_dev, 
        double& normalised_mean, 
        double& normalised_std_dev, 
        shared_ptr<MoThtsEnv> env, 
        shared_ptr<MoThtsDNode> root_node, 
        shared_ptr<MoThtsManager> thts_manager,
        RunID& run_id) 
    {   
        if (run_id.is_python_env()) {
            shared_ptr<EvalPolicy> eval_policy = make_shared<EvalPolicy>(root_node, env, thts_manager);  
            MoPyMCEvaluator evaluator(
                eval_policy, run_id.max_trial_length, thts_manager, run_id.get_env_min_value(), run_id.get_env_max_value());
            evaluator.run_rollouts(run_id.rollouts_per_mc_eval, run_id.eval_threads);
            mean = evaluator.get_mean_mo_ctx_return();
            std_dev = evaluator.get_stddev_mean_mo_ctx_return();
            normalised_mean = evaluator.get_mean_mo_normalised_ctx_return();
            normalised_std_dev = evaluator.get_stddev_mean_mo_normalised_ctx_return();

        } else {
            shared_ptr<EvalPolicy> eval_policy = make_shared<EvalPolicy>(root_node, env, thts_manager);  
            MoMCEvaluator evaluator(
                eval_policy, run_id.max_trial_length, thts_manager, run_id.get_env_min_value(), run_id.get_env_max_value());
            evaluator.run_rollouts(run_id.rollouts_per_mc_eval, run_id.eval_threads);
            mean = evaluator.get_mean_mo_ctx_return();
            std_dev = evaluator.get_stddev_mean_mo_ctx_return();
            normalised_mean = evaluator.get_mean_mo_normalised_ctx_return();
            normalised_std_dev = evaluator.get_stddev_mean_mo_normalised_ctx_return();
        }
    }

    /**
     * Python has a really annoying bug that it crashes when trying to clean up subinterpreters
     * That means that if you ever have >0 subinterpreters, when the number of subinterpreters falls to 0 then it will 
     * crash
     * To get around this, we're going to just run all of our code for 'run_expr' using a subinterpreter itself
     * This way at least one (this one) will always exist throughout the runtime of the program to stop it from crashing
     * LMAO, this didnt even work
     * k python
     * going to try making an subinterpreter that waits to be signalled to exit
    */
    void dummy_subinterpreter_routine(shared_ptr<mutex> m)
    {
        // Setup Py subinterpreter
        // Need to lock with CPython API because pybind11 gil interface not built to work with it
        thts::python::helper::lock_gil();
        PyInterpreterConfig config = {
            .use_main_obmalloc = 0,
            .allow_fork = 0,
            .allow_exec = 0,
            .allow_threads = 1,
            .allow_daemon_threads = 0,
            .check_multi_interp_extensions = 1,
            .gil = PyInterpreterConfig_OWN_GIL,
        };
        PyThreadState *tstate;
        Py_NewInterpreterFromConfig(&tstate, &config);
        if (tstate == NULL) {
            throw runtime_error("Error starting subinterpreter");
        }
        
        // wait until signalled
        m->lock();
        cout << "I am dummy subinterpreter and its frustrating that I exist" << endl;
    }

    shared_ptr<thread> start_dummy_subinterpreter(shared_ptr<mutex> m) 
    {
        return make_shared<thread>(&dummy_subinterpreter_routine, m);
    }

    /**
     * Performs all of the (replicated) runs corresponding to 'run_id'
     * (This is the one exposed function (for now) in run_toy.cpp)
     * 
     * We dont try to gracefully exit
    */
    void run_expr(RunID& run_id) {
        // create results dir + eval file
        create_results_dir(run_id);
        string eval_filename = get_mc_eval_results_filename(run_id);
        ofstream eval_file;
        eval_file.open(eval_filename, ios::out);// | ios::app);
        write_param_header_to_file(run_id, eval_file);
        write_eval_header(eval_file);

        // Run experiment 'replicate' many times
        for (int replicate=0; replicate<run_id.num_repeats; replicate++) {
            // Acquire gil
            shared_ptr<py::gil_scoped_acquire> acq;
            if (run_id.is_python_env()) {
                acq = make_shared<py::gil_scoped_acquire>();
            }

            // print
            cout << "Starting run on " << run_id.env_id << " with alg " << run_id.alg_id << " and params " 
                << helper::unordered_map_pretty_print_string(run_id.alg_params) << ", replicate " << replicate << endl;

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
            shared_ptr<ThtsPool> thts_pool;
            if (run_id.is_python_env()) {
                thts_pool = make_shared<MoPyThtsPool>(thts_manager, root_node, run_id.num_threads);
            } else {
                thts_pool = make_shared<MoThtsPool>(thts_manager, root_node, run_id.num_threads);
            }

            // Release gil during search, let threads grab it as necessary
            shared_ptr<py::gil_scoped_release> rel;
            if (run_id.is_python_env()) {
                rel = make_shared<py::gil_scoped_release>();
            }

            // eval at 0 trials
            double mean, stddev, normalised_mean, normalised_stddev;
            run_mc_eval(
                mean, 
                stddev, 
                normalised_mean, 
                normalised_stddev, 
                env, 
                root_node, 
                thts_manager, 
                run_id);
            write_eval_line(eval_file, replicate, 0.0, 0, mean, stddev, normalised_mean, normalised_stddev);

            // run trials, evaluating every eval delta
            double search_time_elapsed = 0.0;
            while (search_time_elapsed < run_id.search_runtime) {
                thts_pool->run_trials(numeric_limits<int>::max(), run_id.eval_delta);
                search_time_elapsed += run_id.eval_delta;
                run_mc_eval(
                    mean, 
                    stddev, 
                    normalised_mean, 
                    normalised_stddev, 
                    env, 
                    root_node, 
                    thts_manager, 
                    run_id);
                write_eval_line(
                    eval_file, 
                    replicate, 
                    search_time_elapsed, 
                    root_node->get_num_visits(), 
                    mean, 
                    stddev, 
                    normalised_mean, 
                    normalised_stddev);
            }

            // Write tree to file
            if (replicate == 0) {
                shared_ptr<py::gil_scoped_acquire> acq2;
                if (run_id.is_python_env()) {
                    acq2 = make_shared<py::gil_scoped_acquire>();
                }
                string tree_filename = get_tree_filename(run_id, replicate);
                ofstream tree_file;
                tree_file.open(tree_filename, ios::out);
                tree_file << root_node->get_pretty_print_string(4) << endl;
                tree_file.close();
            }

            eval_file.flush();
        }   

        // close eval file
        eval_file.close();
    }

    /**
     * Runs all experiments in 'run_ids'
     * This was in main, until needed to start hacking in the sill subinterpreter bug mitigation
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

        // If running python, make interpreter and immediately run a dummy no-op subinterpreter  
        // (see comment on start_dummy_subinterpreter)
        shared_ptr<py::scoped_interpreter> guard;
        shared_ptr<mutex> dummy_subinterpreter_mutex = make_shared<mutex>();
        shared_ptr<thread> dummy_thread;
        if (need_python) {
            guard = make_shared<py::scoped_interpreter>();
            py::gil_scoped_release rel;
            dummy_subinterpreter_mutex->lock();
            dummy_thread = start_dummy_subinterpreter(dummy_subinterpreter_mutex);
        }

        // Actually run experiments
        for (RunID& run_id : *run_ids) {
            thts::run_expr(run_id);
        }

        // Let the dummy subinterpreter die
        if (need_python) {
            dummy_subinterpreter_mutex->unlock();
            dummy_thread->join();
        }
    }

}