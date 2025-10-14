// #include "main_aux/run_id.h"

// #include "main_aux/run_expr.h"

// #include "helper.h"

// #include "algorithms/uct/uct_manager.h"
// #include "algorithms/uct/hmcts_manager.h"
// #include "algorithms/ments/ments_manager.h"
// #include "algorithms/ments/dents/dents_manager.h"

// #include "algorithms/uct/uct_decision_node.h"
// #include "algorithms/uct/max_uct_decision_node.h"
// #include "algorithms/ments/ments_decision_node.h"
// #include "algorithms/est/est_decision_node.h"
// #include "algorithms/ments/dents/dents_decision_node.h"
// #include "algorithms/uct/hmcts_decision_node.h"
// #include "algorithms/uct/max_uct_decision_node.h"
// #include "algorithms/ments/rents/rents_decision_node.h"
// #include "algorithms/ments/tents/tents_decision_node.h"

// #include "algorithms/common/decaying_temp.h"

// #include "py/pickle_wrapper.h"
// #include "py/py_multiprocessing_thts_env.h"
// #include "py/gym_multiprocessing_thts_env.h"

// #include "main_aux/envs/d_chain.h"
// #include "main_aux/envs/entropy_trap.h"
// #include "main_aux/envs/frozen_lake.h"
// #include "main_aux/envs/sailing.h"

// #include <cmath>

// #include <sstream>
// #include <stdexcept>

// using namespace std;
// using namespace thts;
// using namespace thts::python;

// namespace py = pybind11;

// namespace thts {


//     /**
//      * Hyperparam optimiser - constructor
//      */
//     HyperparamOptimiser::HyperparamOptimiser(
//         string env_id,
//         string expr_id,
//         time_t expr_timestamp,
//         string alg_id,
//         unordered_map<string, pair<double,double>> alg_params_min_max,
//         bool eval_wrt_time,
//         double search_runtime,
//         int max_trial_length,
//         double eval_delta,
//         int rollouts_per_mc_eval,
//         int num_repeats,
//         int num_threads,
//         int eval_threads,
//         bool mcts_mode,
//         bayesopt::Parameters params,
//         ofstream &results_summary_fs,
//         ofstream &results_evals_fs,
//         bool use_std_mean_eval_threshold,
//         double std_mean_eval_threshold) :
//             bayesopt::ContinuousModel(RELEVANT_PARAM_IDS.at(alg_id).size(), params),
//             num_hyperparams(RELEVANT_PARAM_IDS.at(alg_id).size()),
//             env_id(env_id),
//             expr_id(expr_id),
//             expr_timestamp(expr_timestamp),
//             alg_id(alg_id),
//             alg_param_ids(RELEVANT_PARAM_IDS.at(alg_id)),
//             alg_params_min_max(alg_params_min_max),
//             eval_wrt_time(eval_wrt_time),
//             search_runtime(search_runtime),
//             max_trial_length(max_trial_length),
//             eval_delta(eval_delta),
//             rollouts_per_mc_eval(rollouts_per_mc_eval),
//             num_repeats(num_repeats),
//             num_threads(num_threads),
//             eval_threads(eval_threads),
//             num_envs((eval_threads > num_threads) ? eval_threads : num_threads),
//             mcts_mode(mcts_mode),
//             best_eval(numeric_limits<double>::lowest()),
//             best_alg_params(),
//             results_summary_fs(results_summary_fs),
//             results_evals_fs(results_evals_fs),
//             use_std_mean_eval_threshold(use_std_mean_eval_threshold),
//             std_mean_eval_threshold(std_mean_eval_threshold),
//             hp_opt_iter(0)
//     {
//         // error checking
//         if (alg_param_ids.size() != alg_params_min_max.size()) {
//             throw runtime_error("Expecting list of param min/max values to be same size as list of params for alg");
//         }
//         for (string param_id : alg_param_ids) {
//             if (!alg_params_min_max.contains(param_id)) {
//                 stringstream ss;
//                 ss << "Expected list of hyperparams for alg_id=" << alg_id 
//                     << " did not match keys provided in alg_params_min_max. Specifically the param_id=" << param_id 
//                     << " was missing.";
//                 throw runtime_error(ss.str());
//             }
//         }

//         // might as well set bounding box here
//         bayesopt::vectord min_vec(num_hyperparams);
//         bayesopt::vectord max_vec(num_hyperparams);
//         for (size_t i=0; i<alg_param_ids.size(); i++) {
//             pair<double,double> min_max = alg_params_min_max[alg_param_ids[i]];
//             bool use_log_scale = (LOG_SCALE_PARAM_IDS.contains(alg_param_ids[i]));
//             min_vec[i] = use_log_scale ? log(min_max.first) : min_max.first;
//             max_vec[i] = use_log_scale ? log(min_max.second) : min_max.second;
//         }
//         bayesopt::ContinuousModel::setBoundingBox(min_vec,max_vec);
//     };

//     bool HyperparamOptimiser::is_python_env() 
//     {
//         return thts::is_python_env(env_id);
//     }

//     unordered_map<string, double> HyperparamOptimiser::get_alg_params_from_bayesopt_vec(bayesopt::vectord vec)
//     {
//         unordered_map<string, double> alg_params;
//         for (size_t i=0; i<alg_param_ids.size(); i++) {
//             string param_id = alg_param_ids[i];
//             if (BOOLEAN_PARAM_IDS.contains(param_id)) {
//                 pair<double,double> min_max = alg_params_min_max[param_id]; 
//                 alg_params[param_id] = get_bool_val_from_cts_sample(vec[i], min_max.first, min_max.second);
//             } else if (INTEGER_PARAM_IDS.contains(param_id)) {
//                 pair<double,double> min_max = alg_params_min_max[param_id]; 
//                 alg_params[param_id] = get_int_val_from_cts_sample(vec[i], min_max.first, min_max.second);
//             } else {
//                 bool log_scaled = (LOG_SCALE_PARAM_IDS.contains(param_id));
//                 alg_params[param_id] = log_scaled ? exp(vec[i]) : vec[i];
//             }
//         }
//         return alg_params;
//     };

//     bool HyperparamOptimiser::get_bool_val_from_cts_sample(double sample_val, int min, int max)
//     {
//         double midpoint = ((double) min+max) / 2.0;
//         return (sample_val > midpoint);
//     };

//     int HyperparamOptimiser::get_int_val_from_cts_sample(double sample_val, int min, int max)
//     {
//         if (sample_val == max) {
//             return max-1;            
//         }
//         return (int)sample_val;
//     };

//     /**
//      * Helper function to compute mean and std of vector of evals
//      */
//     void compute_mean_and_std_(const vector<double>& evals, double& mean_eval, double& std_eval, double& std_mean_eval)
//     {
//         double evals_sum = 0.0;
//         for (double eval : evals) {
//             evals_sum += eval;
//         }
//         mean_eval = evals_sum / evals.size();

//         double std_eval_sum = 0.0;
//         for (double eval : evals) {
//             std_eval_sum += (eval - mean_eval) * (eval - mean_eval);
//         }
//         std_eval = sqrt(std_eval_sum / (evals.size() - 1));
//         std_mean_eval = std_eval / sqrt(evals.size());
//     }

//     /**
//      * Hyperparam optimiser - fn to optimise
//      * 
//      * If using mean estimate variance threshold, then we keep repeating the params until the variance of the mean 
//      * estimate is below the threshold.
//      * 
//      * If Vbar is the mean estimate, V1 is a rv for value esimate of a run, and std^2=Var(V1)
//      * Then after n repeats, Var(Vbar) = Var(V1) / n approx= std^2 / n
//      * So when std^2 / n < mean_estimate_variance_threshold, then we can stop
//      */
//     double HyperparamOptimiser::evaluateSample(const bayesopt::vectord &query) 
//     {
//         // Run eval on hyperparams
//         unordered_map<string,double> alg_params = get_alg_params_from_bayesopt_vec(query);
//         RunID run_id(
//             env_id,
//             expr_id,
//             expr_timestamp,
//             alg_id,
//             alg_params,
//             eval_wrt_time,
//             search_runtime,
//             eval_delta,
//             rollouts_per_mc_eval,
//             max_trial_length,
//             1, //num_repeats, - now manually running multiple repeats
//             num_threads,
//             eval_threads,
//             mcts_mode
//         );

//         int repeats_run = 0;
//         vector<double> evals;
//         double mean_eval = 0.0;
//         double std_eval = 0.0;
//         double std_mean_eval = 0.0;

//         // run initial repeats
//         while (repeats_run < num_repeats) {
//             double eval = thts::run_expr(run_id, true, repeats_run).at(0);
//             evals.push_back(eval);
//             repeats_run++;
//         }
//         compute_mean_and_std_(evals, mean_eval, std_eval, std_mean_eval);
//         cout << "Hp_opt_iter " << hp_opt_iter << ". mean_eval=" << mean_eval << ",std_mean_eval=" << std_mean_eval << " > " << std_mean_eval_threshold << endl;

//         // While below std threshold, keep running repeats
//         while (use_std_mean_eval_threshold && (std_mean_eval > std_mean_eval_threshold)) {
//             double eval = thts::run_expr(run_id, true, repeats_run).at(0);
//             evals.push_back(eval);
//             repeats_run++;
//             compute_mean_and_std_(evals, mean_eval, std_eval, std_mean_eval);
//             cout << "Hp_opt_iter " << hp_opt_iter << ". mean_eval=" << mean_eval << ",std_mean_eval=" << std_mean_eval << " > " << std_mean_eval_threshold << endl;
//         }

//         // Keep track if this was best hyperparams, and log all repeats, log mean_eval in respective hp_opt files
//         if (mean_eval > best_eval) {
//             best_eval = mean_eval;
//             best_alg_params = alg_params;
//         }
//         write_eval_lines(alg_params, evals);
//         write_summary_line(alg_params, mean_eval);

//         // Remember to increment hp_opt_iter
//         hp_opt_iter++;

//         // bayes opt tried to minimise, so return *-1.0 because want to maximise
//         return -1.0 * mean_eval;
//     };

//     /**
//      * Writes a header with the params for each eval top results_fs
//      */
//     void HyperparamOptimiser::write_header()
//     {
//         // expr params
//         results_summary_fs 
//             << "env_id,alg_id,search_runtime,max_trial_length,rollouts_per_mc_eval,num_repeats,num_threads" << endl;
        
//         results_summary_fs 
//             << env_id << ","
//             << alg_id << ","
//             << search_runtime << ","
//             << max_trial_length << ","
//             << rollouts_per_mc_eval << ","
//             << num_repeats << ","
//             << num_threads 
//             << endl << endl;
        
//         // hyperparams (with sample number (hp_opt_iter) and eval at start/end)
//         results_summary_fs << "hp_opt_iter,";
//         for (string& param_id : alg_param_ids) {
//             results_summary_fs << param_id << ",";
//         } 
//         results_summary_fs << "eval(mc_estimate_expected_utility),best_eval_so_far" << endl;

//         // Print out the min an max params trying
//         results_summary_fs << "MIN,";
//         for (string param_id : alg_param_ids) {
//             results_summary_fs << alg_params_min_max[param_id].first << ",";
//         } 
//         results_summary_fs << "MIN" << endl;
//         results_summary_fs << "MAX,";
//         for (string param_id : alg_param_ids) {
//             results_summary_fs << alg_params_min_max[param_id].second << ",";
//         } 
//         results_summary_fs << "MAX" << endl;
        
//         // for evals fs, just want the list of hyperparms, and the eval
//         results_evals_fs << "hp_opt_iter,replicate,";
//         for (string& param_id : alg_param_ids) {
//             results_evals_fs << param_id << ",";
//         } 
//         results_evals_fs << "eval(mc_estimate_expected_utility)" << endl;
//     };

//     /**
//      * Write eval/hyperparam sample line to file
//      * - note that hp_opt_iter only used but updated
//      */
//     void HyperparamOptimiser::write_eval_lines(unordered_map<string,double> alg_params, vector<double>& evals)
//     {   
//         for (size_t i=0; i<evals.size(); i++) {
//             results_evals_fs << hp_opt_iter << "," << i << ",";
//             for (string param_id : alg_param_ids) {
//                 results_evals_fs << alg_params[param_id] << ",";
//             }
//             results_evals_fs << evals[i] << endl;
//         }
//     };

//     /**
//      * Write hyperparam sample line to file
//      * - note that hp_opt_iter only used here, and also updated here
//      */
//     void HyperparamOptimiser::write_summary_line(unordered_map<string,double> alg_params, double mean_eval)
//     {   
//         results_summary_fs << hp_opt_iter << ",";
//         for (string param_id : alg_param_ids) {
//             results_summary_fs << alg_params[param_id] << ",";
//         }
//         results_summary_fs << mean_eval << "," << best_eval << endl;
//     };

//     void HyperparamOptimiser::write_best_eval()
//     {
//         results_summary_fs << endl;
//         results_summary_fs << "Best eval with params:" << endl;
//         results_summary_fs << "eval (mc_estimate_expected_utility) = " << best_eval << endl;
//         for (pair<string,double> pr : best_alg_params) {
//             results_summary_fs << pr.first << " = " << pr.second << endl;
//         }
//     };

//     /**
//      * Gets hyperparam optimiser from expr_id
//      */
//     shared_ptr<HyperparamOptimiser> get_hyperparam_optimiser_from_expr_id(
//         string expr_id, time_t expr_timestamp, ofstream &hp_opt_summary_fs, ofstream &hp_opt_evals_fs)
//     {
//         // Params shared across optimisations (related to envs / hp_opt, and not algs themselves)
//         string env_id = HP_OPT_EXPR_ID_TO_ENV_ID.at(expr_id);
//         bool eval_wrt_time = false;
//         double search_runtime = 10000.0;
//         int max_trial_length = ENV_ID_MAX_TRIAL_LEN.at(env_id);
//         double eval_delta = 10000.0; 
//         int rollouts_per_mc_eval = 1024;
//         int num_repeats = 10; // min repeats
//         int num_threads = 16;
//         int eval_threads = 16;
//         bool use_std_mean_eval_threshold = true;

//         // Params being tuned
//         string alg_id;
//         unordered_map<string, pair<double,double>> alg_params_min_max;

//         // Defualt Q values and std_mean_eval_thresholds (default values are for sparse rewards on frozen lake envs)
//         double min_default_q_value = 0.0;
//         // if (env_id == FROZEN_LAKE_D_8x8_ENV_ID || env_id == SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID) {
//         //     min_default_q_value = -((double) max_trial_length);
//         // } else if (env_id == SAILING_ENV_NORTH_ID || env_id == SAILING_ENV_SOUTH_EAST_ID) {
//         //     min_default_q_value = -5.0 * ((double) max_trial_length);
//         // } 

//         // std mean eval thresholds
//         double std_mean_eval_threshold = 1.0; 
//         if (env_id == FROZEN_LAKE_D_8x8_ENV_ID) {
//             std_mean_eval_threshold = 1.0;
//         } else if (env_id == FROZEN_LAKE_S_8x8_ENV_ID) {
//             std_mean_eval_threshold = 0.015;
//         } else if (env_id == SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID) {
//             std_mean_eval_threshold = 0.05;
//         } else if (env_id == SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID) {
//             std_mean_eval_threshold = 0.005; 
//         } else if (env_id == SAILING_ENV_NORTH_ID) {
//             std_mean_eval_threshold = 0.5;
//         } else if (env_id == SAILING_ENV_SOUTH_EAST_ID) {
//             std_mean_eval_threshold = 0.5; 
//         }

//         bool mcts_mode = (env_id == SAILING_ENV_NORTH_ID || env_id == SAILING_ENV_SOUTH_EAST_ID);

//         // UCT
//         if (expr_id == HP_OPT_600_UCT_EXPR_ID 
//             || expr_id == HP_OPT_601_UCT_EXPR_ID
//             || expr_id == HP_OPT_602_UCT_EXPR_ID
//             || expr_id == HP_OPT_603_UCT_EXPR_ID
//             || expr_id == HP_OPT_604_UCT_EXPR_ID
//             || expr_id == HP_OPT_605_UCT_EXPR_ID) 
//         {
//             alg_id = UCT_ALG_ID;
//             alg_params_min_max = {
//                 {ADAPTIVE_BIAS_PARAM_ID, make_pair(0.0, 1.0)},
//                 {BIAS_PARAM_ID, make_pair(0.001, 1000.0)},
//             };
//         }
//         // MaxUCT
//         else if (expr_id == HP_OPT_610_MAX_UCT_EXPR_ID
//             || expr_id == HP_OPT_611_MAX_UCT_EXPR_ID
//             || expr_id == HP_OPT_612_MAX_UCT_EXPR_ID
//             || expr_id == HP_OPT_613_MAX_UCT_EXPR_ID
//             || expr_id == HP_OPT_614_MAX_UCT_EXPR_ID
//             || expr_id == HP_OPT_615_MAX_UCT_EXPR_ID)
//         {
//             alg_id = MAX_UCT_ALG_ID;
//             alg_params_min_max = {
//                 {ADAPTIVE_BIAS_PARAM_ID, make_pair(0.0, 1.0)},
//                 {BIAS_PARAM_ID, make_pair(0.001, 1000.0)},
//             };
//         }
//         // MENTS
//         else if (expr_id == HP_OPT_620_MENTS_EXPR_ID
//             || expr_id == HP_OPT_621_MENTS_EXPR_ID
//             || expr_id == HP_OPT_622_MENTS_EXPR_ID
//             || expr_id == HP_OPT_623_MENTS_EXPR_ID
//             || expr_id == HP_OPT_624_MENTS_EXPR_ID
//             || expr_id == HP_OPT_625_MENTS_EXPR_ID)
//         {   
//             alg_id = MENTS_ALG_ID;
//             alg_params_min_max = {
//                 {NORMALISE_Q_VALUES_PARAM_ID, make_pair(0.0, 1.0)},
//                 {TEMP_PARAM_ID, make_pair(0.001, 1000.0)},
//                 {EPSILON_PARAM_ID, make_pair(0.000001, 1.0)},
//                 // {DEFAULT_Q_VALUE_PARAM_ID, make_pair(min_default_q_value,0.0)}
//                 {DEFAULT_Q_VALUE_PARAM_ID, make_pair(min_default_q_value,min_default_q_value)}
//             };
//         }
//         // BTS
//         else if (expr_id == HP_OPT_630_BTS_EXPR_ID
//             || expr_id == HP_OPT_631_BTS_EXPR_ID
//             || expr_id == HP_OPT_632_BTS_EXPR_ID
//             || expr_id == HP_OPT_633_BTS_EXPR_ID
//             || expr_id == HP_OPT_634_BTS_EXPR_ID
//             || expr_id == HP_OPT_635_BTS_EXPR_ID)
//         {
//             alg_id = BTS_ALG_ID;
//             alg_params_min_max = {
//                 {NORMALISE_Q_VALUES_PARAM_ID, make_pair(0.0, 1.0)},
//                 {TEMP_PARAM_ID, make_pair(0.001, 1000.0)},
//                 {DECAY_FN_PARAM_ID, make_pair(0.0, 3.0)},
//                 {DECAY_FN_SCALE_PARAM_ID, make_pair(0.01, 100.0)},
//                 {EPSILON_PARAM_ID, make_pair(0.000001, 1.0)},
//                 // {DEFAULT_Q_VALUE_PARAM_ID, make_pair(min_default_q_value,0.0)}
//                 {DEFAULT_Q_VALUE_PARAM_ID, make_pair(min_default_q_value,min_default_q_value)}
//             };
//         }
//         // DENTS
//         else if (expr_id == HP_OPT_640_DENTS_EXPR_ID
//             || expr_id == HP_OPT_641_DENTS_EXPR_ID
//             || expr_id == HP_OPT_642_DENTS_EXPR_ID
//             || expr_id == HP_OPT_643_DENTS_EXPR_ID
//             || expr_id == HP_OPT_644_DENTS_EXPR_ID
//             || expr_id == HP_OPT_645_DENTS_EXPR_ID)
//         {
//             alg_id = DENTS_ALG_ID;
//             alg_params_min_max = {
//                 {NORMALISE_Q_VALUES_PARAM_ID, make_pair(0.0, 1.0)},
//                 {TEMP_PARAM_ID, make_pair(0.001, 1000.0)},
//                 {DECAY_FN_PARAM_ID, make_pair(0.0, 3.0)},
//                 {DECAY_FN_SCALE_PARAM_ID, make_pair(0.01, 100.0)},
//                 {ENTROPY_COEFF_PARAM_ID, make_pair(0.001, 1000.0)},
//                 {ENTROPY_DECAY_FN_PARAM_ID, make_pair(0.0, 3.0)},
//                 {ENTROPY_DECAY_FN_SCALE_PARAM_ID, make_pair(0.01, 100.0)},
//                 {EPSILON_PARAM_ID, make_pair(0.000001, 1.0)},
//                 // {DEFAULT_Q_VALUE_PARAM_ID, make_pair(min_default_q_value,0.0)}
//                 {DEFAULT_Q_VALUE_PARAM_ID, make_pair(min_default_q_value,min_default_q_value)}
//             };
//         }
//         // RENTS
//         else if (expr_id == HP_OPT_650_RENTS_EXPR_ID
//             || expr_id == HP_OPT_651_RENTS_EXPR_ID
//             || expr_id == HP_OPT_652_RENTS_EXPR_ID
//             || expr_id == HP_OPT_653_RENTS_EXPR_ID
//             || expr_id == HP_OPT_654_RENTS_EXPR_ID
//             || expr_id == HP_OPT_655_RENTS_EXPR_ID)
//         {
//             alg_id = RENTS_ALG_ID;
//             alg_params_min_max = {
//                 {NORMALISE_Q_VALUES_PARAM_ID, make_pair(0.0, 1.0)},
//                 {TEMP_PARAM_ID, make_pair(0.001, 1000.0)},
//                 {EPSILON_PARAM_ID, make_pair(0.000001, 1.0)},
//                 // {DEFAULT_Q_VALUE_PARAM_ID, make_pair(min_default_q_value,0.0)}
//                 {DEFAULT_Q_VALUE_PARAM_ID, make_pair(min_default_q_value,min_default_q_value)}
//             };
//         }
//         // TENTS
//         else if (expr_id == HP_OPT_660_TENTS_EXPR_ID
//             || expr_id == HP_OPT_661_TENTS_EXPR_ID
//             || expr_id == HP_OPT_662_TENTS_EXPR_ID
//             || expr_id == HP_OPT_663_TENTS_EXPR_ID
//             || expr_id == HP_OPT_664_TENTS_EXPR_ID
//             || expr_id == HP_OPT_665_TENTS_EXPR_ID)
//         {
//             alg_id = TENTS_ALG_ID;
//             alg_params_min_max = {
//                 {NORMALISE_Q_VALUES_PARAM_ID, make_pair(0.0, 1.0)},
//                 {TEMP_PARAM_ID, make_pair(0.001, 1000.0)},
//                 {EPSILON_PARAM_ID, make_pair(0.000001, 1.0)},
//                 // {DEFAULT_Q_VALUE_PARAM_ID, make_pair(min_default_q_value,0.0)}
//                 {DEFAULT_Q_VALUE_PARAM_ID, make_pair(min_default_q_value,min_default_q_value)}
//             };
//         }
//         // HMCTS
//         else if (expr_id == HP_OPT_670_HMCTS_EXPR_ID
//             || expr_id == HP_OPT_671_HMCTS_EXPR_ID
//             || expr_id == HP_OPT_672_HMCTS_EXPR_ID
//             || expr_id == HP_OPT_673_HMCTS_EXPR_ID
//             || expr_id == HP_OPT_674_HMCTS_EXPR_ID
//             || expr_id == HP_OPT_675_HMCTS_EXPR_ID)
//         {
//             alg_id = HMCTS_ALG_ID;
//             alg_params_min_max = {
//                 {ADAPTIVE_BIAS_PARAM_ID, make_pair(0.0, 1.0)},
//                 {BIAS_PARAM_ID, make_pair(0.001, 1000.0)},
//                 {UCT_BUDGET_PARAM_ID, make_pair(1.0, 5000.0)},
//             };
//         }
//         // Default, haven't set up hp opt experiments for this env
//         else 
//         { 
//             stringstream ss;
//             ss << "Error in get_hyperparam_optimiser_from_expr_id for expr_id = " << expr_id;
//             throw runtime_error(ss.str());
//         }

//         // Bayesopt params
//         bayesopt::Parameters bo_params;
//         bo_params.surr_name = "sGaussianProcessML";
//         bo_params.noise = std_mean_eval_threshold*std_mean_eval_threshold; //1.0; 
//         bo_params.n_iterations = 190;
//         bo_params.n_init_samples = 10;
//         bo_params.n_iter_relearn = 10;
//         bo_params.verbose_level = 0;

//         return make_shared<HyperparamOptimiser>(
//             env_id,
//             expr_id,
//             expr_timestamp,
//             alg_id,
//             alg_params_min_max,
//             eval_wrt_time,
//             search_runtime,
//             max_trial_length,
//             eval_delta,
//             rollouts_per_mc_eval,
//             num_repeats,
//             num_threads,
//             eval_threads,
//             mcts_mode,
//             bo_params, 
//             hp_opt_summary_fs,
//             hp_opt_evals_fs,
//             use_std_mean_eval_threshold,
//             std_mean_eval_threshold
//         );
//     };

//     /**
//      * Lookup expr_id from prefix
//      */
//     string lookup_expr_id_from_prefix(string expr_id_prefix) 
//     {
//         for (const string& expr_id : ALL_EXPR_IDS) {
//             if (expr_id.starts_with(expr_id_prefix)) {
//                 return expr_id;
//             }
//         }
//         throw runtime_error(
//             "Error looking up expr_id from prefix. Either forgot to add expr_id to 'ALL_EXPR_IDS' list or typo?");
//     }
    
//     /**
//      * Helper to make a string of:
//      * "param1=val1/param2=val2/.../paramN=valN/"
//      * Old version output:
//      * "param1=val1,param2=val2,...,paramN=valN",
//      * but lead to filenames that were too long
//     */
//     string get_params_string_helper(RunID& run_id) {
//         stringstream ss;
//         const vector<string> &relevant_param_ids = RELEVANT_PARAM_IDS.at(run_id.alg_id);
//         unordered_set<string> relevant_param_ids_set(relevant_param_ids.begin(),relevant_param_ids.end());
//         for (pair<string,double> param_val_entry : run_id.alg_params) {
//             if (!relevant_param_ids_set.contains(param_val_entry.first)) {
//                 continue;
//             }
//             ss << param_val_entry.first << "=" << param_val_entry.second << "/";
//         }
//         return ss.str();
//     }

//     /**
//      * Gets the results directory for this run (doesn't check/make)
//     */
//     string get_results_dir(RunID& run_id) {
//         stringstream ss;
//         ss << "results_aux/" 
//             << run_id.expr_id << "_" << run_id.expr_timestamp << "/" 
//             << run_id.env_id << "/" 
//             << run_id.alg_id << "/"
//             << get_params_string_helper(run_id);
//         return ss.str();
//     }
    
//     /**
//      * Checks if env corresponding to 'env_id' is a python env
//      */
//     bool is_python_env(string env_id) 
//     {
//         return (PY_ENVS.contains(env_id) 
//             || GYM_ENVS.contains(env_id));
//     }

//     /**
//      * Create and return the env
//     */
//     shared_ptr<ThtsEnv> get_env(RunID& run_id) 
//     {
//         string thts_unique_filename = get_results_dir(run_id);
//         string& env_id = run_id.env_id;

//         if (GYM_ENVS.contains(env_id)) {
//             shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
//             return make_shared<GymMultiprocessingThtsEnv>(pickle_wrapper, thts_unique_filename, env_id);
//         }

//         if (env_id == D_CHAIN_10_ENV_ID)
//         {
//             return make_shared<DChainEnv>(10,1.0);
//         }
//         if (env_id == MOD_D_CHAIN_10_ENV_ID)
//         {
//             return make_shared<DChainEnv>(10,0.5); 
//         }

//         if (env_id == ENTROPY_TRAP_10_ENV_ID)
//         {
//             return make_shared<EntropyTrapEnv>(10,10,1.0);
//         }

//         if (env_id == ENTROPY_TRAP_15_ENV_ID)
//         {
//             return make_shared<EntropyTrapEnv>(15,15,1.0);
//         }

//         if (env_id == FROZEN_LAKE_NO_HOLE_DENSE_ENV_ID || env_id == FROZEN_LAKE_NO_HOLE_SPARSE_LEN_ENV_ID || env_id == FROZEN_LAKE_NO_HOLE_SPARSE_DISCOUNTED_ENV_ID)
//         {
//             int reward_type = FL_DENSE_REWARD;
//             if (env_id == FROZEN_LAKE_NO_HOLE_SPARSE_LEN_ENV_ID) {
//                 reward_type = FL_SPARSE_LEN_REWARD;
//             } else if (env_id == FROZEN_LAKE_NO_HOLE_SPARSE_DISCOUNTED_ENV_ID) {
//                 reward_type = FL_SPARSE_DISCOUNTED_REWARD;
//             }
//             return make_shared<FrozenLakeEnv>(6,6,FL_6x6_NO_HOLE_MAP,false,reward_type);
//         }

//         if (env_id == FROZEN_LAKE_D_8x8_ENV_ID)
//         {
//             return make_shared<FrozenLakeEnv>(8,8,FL_8x8_MAP,false,FL_DENSE_REWARD);
//         }
//         if (env_id == FROZEN_LAKE_S_8x8_ENV_ID)
//         {
//             return make_shared<FrozenLakeEnv>(8,8,FL_8x8_MAP,false,FL_SPARSE_DISCOUNTED_REWARD);
//         }

//         if (env_id == FROZEN_LAKE_D_8x16_ENV_ID)
//         {
//             return make_shared<FrozenLakeEnv>(8,16,FL_GEN_8x16_MAP,false,FL_DENSE_REWARD);
//         }
//         if (env_id == FROZEN_LAKE_S_8x16_ENV_ID)
//         {
//             return make_shared<FrozenLakeEnv>(8,16,FL_GEN_8x16_MAP,false,FL_SPARSE_DISCOUNTED_REWARD);
//         }

//         if (env_id == FROZEN_LAKE_D_16x16_ENV_ID)
//         {
//             return make_shared<FrozenLakeEnv>(8,16,FL_GEN_16x16_MAP,false,FL_DENSE_REWARD);
//         }
//         if (env_id == FROZEN_LAKE_S_16x16_ENV_ID)
//         {
//             return make_shared<FrozenLakeEnv>(8,16,FL_GEN_16x16_MAP,false,FL_SPARSE_DISCOUNTED_REWARD);
//         }

//         if (env_id == SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID)
//         {
//             return make_shared<FrozenLakeEnv>(4,4,FL_4x4_MAP,true,FL_DENSE_REWARD);
//         }
//         if (env_id == SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID)
//         {
//             return make_shared<FrozenLakeEnv>(4,4,FL_4x4_MAP,true,FL_SPARSE_DISCOUNTED_REWARD,1.0);
//         }

//         if (env_id == SLIPPY_FROZEN_LAKE_D_5x5_ENV_ID)
//         {
//             return make_shared<FrozenLakeEnv>(5,5,FL_GEN_5x5_MAP,true,FL_DENSE_REWARD);
//         }
//         if (env_id == SLIPPY_FROZEN_LAKE_S_5x5_ENV_ID)
//         {
//             return make_shared<FrozenLakeEnv>(5,5,FL_GEN_5x5_MAP,true,FL_SPARSE_DISCOUNTED_REWARD,1.0);
//         }

//         if (env_id == SLIPPY_FROZEN_LAKE_D_6x6_ENV_ID)
//         {
//             return make_shared<FrozenLakeEnv>(5,5,FL_GEN_6x6_MAP,true,FL_DENSE_REWARD);
//         }
//         if (env_id == SLIPPY_FROZEN_LAKE_S_6x6_ENV_ID)
//         {
//             return make_shared<FrozenLakeEnv>(5,5,FL_GEN_6x6_MAP,true,FL_SPARSE_DISCOUNTED_REWARD,1.0);
//         }

//         if (env_id == SAILING_ENV_NORTH_ID)
//         {
//             return make_shared<SailingEnv>(8,8,NN);
//         }
        
//         if (env_id == SAILING_ENV_SOUTH_EAST_ID)
//         {
//             return make_shared<SailingEnv>(8,8,SE);
//         }

//         if (env_id == SAILING_8x16_ENV_NORTH_ID)
//         {
//             return make_shared<SailingEnv>(8,16,NN);
//         }
        
//         if (env_id == SAILING_8x16_ENV_SOUTH_EAST_ID)
//         {
//             return make_shared<SailingEnv>(8,16,SE);
//         }

//         if (env_id == SAILING_16x16_ENV_NORTH_ID)
//         {
//             return make_shared<SailingEnv>(16,16,NN);
//         }
        
//         if (env_id == SAILING_16x16_ENV_SOUTH_EAST_ID)
//         {
//             return make_shared<SailingEnv>(16,16,SE);
//         }

//         stringstream ss;
//         ss << "Error in get_env for env_id = " << env_id;
//         throw runtime_error(ss.str());
//     }

// }