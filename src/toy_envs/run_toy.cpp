#include "toy_envs/run_toy.h"

#include "mc_eval.h"
#include "thts.h"
#include "helper_templates.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

namespace thts {
    /**
     * Gets the results directory for this run (doesn't check/make)
    */
    string get_results_dir(RunID& run_id) {
        stringstream ss;
        ss << "results/" 
            << run_id.env_id << "/" 
            << run_id.env_instance_id << "/"
            << run_id.expr_id << "/" 
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
            << "eval_"
            << get_params_string_helper(run_id)
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
     * Returns the filename for the mc eval results file
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
        eval_out_file << "replicate,num_trials,mc_eval_mean,mc_eval_std" << endl;
    }

    /**
     * Write the results from an evaluation to the eval file
    */
    void write_eval_line(
        ofstream& eval_out_file, int replicate, int num_trials, double mc_eval_mean, double mc_eval_std) 
    {
        eval_out_file 
            << replicate << ","
            << num_trials << ","
            << mc_eval_mean << ","
            << mc_eval_std << endl;
    }

    /**
     * Perform an mc eval
     * Returns the mean and std via the double& values
    */
    void run_mc_eval(
        double& mean, 
        double& std_dev, 
        shared_ptr<ThtsEnv> env, 
        shared_ptr<ThtsDNode> root_node, 
        shared_ptr<ThtsManager> thts_manager,
        int max_trial_length,
        int eval_rollouts,
        int eval_threads) 
    {   
        EvalPolicy eval_policy(root_node, env, *thts_manager);  
        MCEvaluator evaluator(env, eval_policy, max_trial_length, *thts_manager);
        evaluator.run_rollouts(eval_rollouts, eval_threads);
        mean = evaluator.get_mean_return();
        std_dev = evaluator.get_stddev_return();
    }

    /**
     * Performs all of the (replicated) runs corresponding to 'run_id'
     * (This is the one exposed function (for now) in run_toy.cpp)
    */
    void perform_toy_env_runs(RunID& run_id) {
        // create results dir + eval file
        create_results_dir(run_id);
        string eval_filename = get_mc_eval_results_filename(run_id);
        ofstream eval_file;
        eval_file.open(eval_filename, ios::out);// | ios::app);
        write_param_header_to_file(run_id, eval_file);
        write_eval_header(eval_file);

        for (int replicate=0; replicate<run_id.num_repeats; replicate++) {
            // print
            cout << "Starting run on " << run_id.env_instance_id << " with alg " << run_id.alg_id << " and params " 
                << helper::unordered_map_pretty_print_string(run_id.alg_params) << ", replicate " << replicate << endl;

            // setup env
            shared_ptr<ThtsEnv> env = run_id.get_env();
            shared_ptr<ThtsManager> thts_manager = run_id.get_thts_manager(env);
            shared_ptr<ThtsDNode> root_node = run_id.get_root_search_node(env, thts_manager);
            shared_ptr<ThtsLogger> logger = run_id.get_logger();
            {
            ThtsPool thts_pool(thts_manager, root_node, run_id.num_threads, logger);

            // eval at 0 trials
            double mean, stddev;
            run_mc_eval(
                mean, 
                stddev, 
                env, 
                root_node, 
                thts_manager, 
                run_id.max_trial_length, 
                run_id.rollouts_per_mc_eval, 
                run_id.eval_threads);
            write_eval_line(eval_file, replicate, 0, mean, stddev);

            // run trials, evaluating every eval delta
            int trials_run = 0;
            while (trials_run < run_id.num_trials) {
                thts_pool.run_trials(run_id.mc_eval_trials_delta);
                trials_run += run_id.mc_eval_trials_delta;
                run_mc_eval(
                    mean, 
                    stddev, 
                    env, 
                    root_node, 
                    thts_manager, 
                    run_id.max_trial_length, 
                    run_id.rollouts_per_mc_eval, 
                    run_id.eval_threads);
                write_eval_line(eval_file, replicate, trials_run, mean, stddev);
            }

            // Create + write logger output to file
            string logger_filename = get_logger_results_filename(run_id, replicate);
            ofstream logger_file;
            logger_file.open(logger_filename, ios::out);// | ios::app);
            write_param_header_to_file(run_id, logger_file);
            logger->write_to_ostream(logger_file);
            logger_file.close();

            // Write tree to file
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
}