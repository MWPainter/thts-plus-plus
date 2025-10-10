






// MAKE HpoptConfigValue
// AND HpoptConfigMap
// THEN define debug config for hp_opt
// THEN start building implementations
// 1. first just construct the manager objects and validate the debug configs
// 2. then build out experiment loops
// 3. then run and test the debug and hp_opt loops
// 4. move config data into config.h files
// 5. start running and finishing off experiments!









// #pragma once

// #include "thts_env.h"
// #include "thts_manager.h"
// #include "thts_decision_node.h"

// #include <ctime>
// #include <fstream>
// #include <memory>
// #include <string>
// #include <tuple>
// #include <unordered_map>
// #include <unordered_set>

// #include "bayesopt/bayesopt.hpp"
// #include "bayesopt/parameters.hpp"

// #include "main_aux/configs/xpr_config.h"


// namespace thts {
//     /**
//      * Class for running hyperparam optimisation
//      * 
//      * 'alg_param_ids' 
//      *      is used to map between vectors (used in bayesopt) and param ids
//      * 'alg_param_min_max[param_id]' 
//      *      specifies the maximum and minimum values to use in bayesopt for param id
//      *      N.B. min and max can be arbitrary for a boolean value, but may as well be 0.0, and 1.0
//      *          and for integer value, we will sample in the *integer* range [min,max)
//      */
//     class HyperparamOptimiser : public bayesopt::ContinuousModel
//     {
//         public:
//             int num_hyperparams;

//             std::string env_id;
//             std::string expr_id;
//             std::time_t expr_timestamp;
//             std::string alg_id;

//             std::vector<std::string> alg_param_ids;
//             std::unordered_map<std::string, std::pair<double,double>> alg_params_min_max;

//             bool eval_wrt_time;
//             double search_runtime;
//             int max_trial_length;
//             double eval_delta;
//             int rollouts_per_mc_eval;
//             int num_repeats;
//             int num_threads;
//             int eval_threads;
//             int num_envs;

//             bool mcts_mode;

//             double best_eval;
//             std::unordered_map<std::string, double> best_alg_params;

//             std::ofstream &results_summary_fs;
//             std::ofstream &results_evals_fs;

//             bool use_std_mean_eval_threshold;
//             double std_mean_eval_threshold;

//             int hp_opt_iter;
            
//             HyperparamOptimiser(
//                 std::string env_id,
//                 std::string expr_id,
//                 std::time_t expr_timestamp,
//                 std::string alg_id,
//                 std::unordered_map<std::string, std::pair<double,double>> alg_params_min_max,
//                 bool eval_wrt_time,
//                 double search_runtime,
//                 int max_trial_length,
//                 double eval_delta,
//                 int rollouts_per_mc_eval,
//                 int num_repeats,
//                 int num_threads,
//                 int eval_threads,
//                 bool mcts_mode,
//                 bayesopt::Parameters params,
//                 std::ofstream &results_summary_fs,
//                 std::ofstream &results_evals_fs,
//                 bool use_std_mean_eval_threshold=false,
//                 double std_mean_eval_threshold=0.0);

//             bool is_python_env();

//             virtual std::unordered_map<std::string, double> get_alg_params_from_bayesopt_vec(bayesopt::vectord vec);

//             bool get_bool_val_from_cts_sample(double sample_val, int min, int max);

//             int get_int_val_from_cts_sample(double sample_val, int min, int max);

//             virtual double evaluateSample(const bayesopt::vectord &query) override;

//             void write_header();

//         private:
//             void write_eval_lines(std::unordered_map<std::string,double> alg_params, std::vector<double>& evals);
//             void write_summary_line(std::unordered_map<std::string,double> alg_params, double mean_eval);

//         public:
//             void write_best_eval();
//     };

//     /**
//      * Creates and returns a hyperparamters optimiser from experiment id
//     */
//     std::shared_ptr<HyperparamOptimiser> get_hyperparam_optimiser_from_expr_id(
//         std::string expr_id, std::time_t expr_timestamp, std::ofstream &hp_opt_summary_fs, std::ofstream &hp_opt_evals_fs);

//     /**
//      * Lookup expr_id from prefix
//      */
//     std::string lookup_expr_id_from_prefix(std::string expr_id_prefix);

//     /**
//      * A unique results directory for each RunID
//      */
//     std::string get_results_dir(RunID& run_id);

//     /**
//      * Checks if env corresponding to 'env_id' is a python env
//      */
//     bool is_python_env(std::string env_id);

//     /**
//      * Create the env corresponding to 'env_id' and return is
//      */
//     std::shared_ptr<ThtsEnv> get_env(RunID& run_id);
// }