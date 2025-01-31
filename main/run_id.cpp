#include "main/run_id.h"

#include "main/run_expr.h"

#include "mo/chmcts_manager.h"
#include "mo/czt_manager.h"
#include "mo/smt_bts_manager.h"
#include "mo/smt_dents_manager.h"

#include "mo/czt_decision_node.h"
#include "mo/chmcts_decision_node.h"
#include "mo/smt_bts_decision_node.h"
#include "mo/smt_dents_decision_node.h"

#include "py/pickle_wrapper.h"
#include "py/mo_gym_multiprocessing_thts_env.h"
#include "py/timed_mo_gym_multiprocessing_thts_env.h"

#include "test/mo/test_mo_thts_env.h"
#include "py/mo_py_multiprocessing_thts_env.h"

#include <sstream>
#include <stdexcept>

using namespace std;
using namespace thts;
using namespace thts::python;
using namespace thts::test;
using namespace pybind11::literals;

namespace py = pybind11;

namespace thts {

    /**
     * Default constructor
    */
    RunID::RunID() {}

    /**
     * Initialised constructor
    */
    RunID::RunID(
        string env_id,
        string expr_id,
        time_t expr_timestamp,
        string alg_id,
        unordered_map<string, double>& alg_params,
        bool eval_wrt_time,
        double search_runtime,
        int max_trial_length,
        double eval_delta,
        int rollouts_per_mc_eval,
        int num_repeats,
        int num_threads,
        int eval_threads) :
            env_id(env_id),
            expr_id(expr_id),
            expr_timestamp(expr_timestamp),
            alg_id(alg_id),
            alg_params(alg_params),
            bias(UctManagerArgs::bias_default),
            temp(BtsManagerArgs::temp_default),
            decay_fn(DECAY_FN_CONST),
            decay_fn_coeff(1.0),
            decay_fn_scale(1.0),
            eval_wrt_time(eval_wrt_time),
            search_runtime(search_runtime),
            max_trial_length(max_trial_length),
            eval_delta(eval_delta),
            rollouts_per_mc_eval(rollouts_per_mc_eval),
            num_repeats(num_repeats),
            num_threads(num_threads),
            eval_threads(eval_threads),
            num_envs((eval_threads > num_threads) ? eval_threads : num_threads)
    {
        if (alg_params.contains(BIAS_PARAM_ID)) {
            bias = alg_params[BIAS_PARAM_ID];
        }
        if (alg_params.contains(TEMP_PARAM_ID)) {
            temp = alg_params[TEMP_PARAM_ID];
        }
        if (alg_params.contains(DECAY_FN_PARAM_ID)) {
            decay_fn = alg_params[DECAY_FN_PARAM_ID];
        }
        if (alg_params.contains(DECAY_FN_COEFF_PARAM_ID)) {
            decay_fn_coeff = alg_params[DECAY_FN_COEFF_PARAM_ID];
        }
        if (alg_params.contains(DECAY_FN_SCALE_PARAM_ID)) {
            decay_fn_scale = alg_params[DECAY_FN_SCALE_PARAM_ID];
        }
    }

    string RunID::get_results_dir() 
    {
        return thts::get_results_dir(*this);
    }

    bool RunID::is_python_env() 
    {
        return thts::is_python_env(env_id);
    }

    shared_ptr<MoThtsEnv> RunID::get_env() 
    {
        return thts::get_env(*this);
    }

    /**
     * Create thts manager
    */
    shared_ptr<MoThtsManager> RunID::get_thts_manager(shared_ptr<MoThtsEnv> env) 
    {
        if (alg_id == UCT_ALG_ID) {
            UctManagerArgs manager_args(env);
            manager_args.max_depth = max_trial_length;
            manager_args.mcts_mode = false;
            manager_args.num_threads = num_threads;
            manager_args.num_envs = num_envs;
            manager_args.bias = bias;
            return make_shared<UctManager>(manager_args);
        }

        if (alg_id == BTS_ALG_ID) {
            BtsManagerArgs manager_args(env);
            manager_args.max_depth = max_trial_length;
            manager_args.mcts_mode = false;
            manager_args.num_threads = num_threads;
            manager_args.num_envs = num_envs;
            manager_args.temp = temp;
            // TODO: handle the decay fn stuff (copy from xpr_go?)
            return make_shared<BtsManager>(manager_args);
        }
        
        // TODO: add args for toher algs

        stringstream ss;
        ss << "Error in RunID get_thts_manager for alg_id = " << alg_id;
        throw runtime_error(ss.str());
    }

    /**
     * Return a root search node
    */
    shared_ptr<MoThtsDNode> RunID::get_root_search_node(shared_ptr<MoThtsEnv> env, shared_ptr<MoThtsManager> manager) 
    {
        if (alg_id == UCT_ALG_ID) {
            shared_ptr<UcttManager> uct_manager = static_pointer_cast<UctManager>(manager);
            return make_shared<UctDNode>(uct_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == BTS_ALG_ID) {
            shared_ptr<BtsManager> bts_manager = static_pointer_cast<BtsManager>(manager);
            return make_shared<BtsDNode>(bts_manager, env->get_initial_state_itfc(), 0, 0);
        }
        // TODO: include other algs

        stringstream ss;
        ss << "Error in RunID get_root_search_node for alg_id = " << alg_id;
        throw runtime_error(ss.str());
    }

    /**
     * Gets a list of RunID objects from a given expr id
    */
    shared_ptr<vector<RunID>> get_run_ids_from_expr_id_prefix(string expr_id_prefix) 
    {   
        string expr_id = lookup_expr_id_from_prefix(expr_id_prefix);
        shared_ptr<vector<RunID>> run_ids = make_shared<vector<RunID>>();

        // TODO: define RunId's for experiments

        // // expr_id: 000_debug 
        // // debug expr id for debugging
        // if (expr_id == DEBUG_EXPR_ID) {
        //     string env_id = DST_ENV_ID;
        //     // string env_id = DEBUG_PY_ENV_1_ID;
        //     time_t expr_timestamp = std::time(nullptr);
        //     double search_runtime = 1.0;
        //     int max_trial_length = 50;
        //     double eval_delta = 0.5;
        //     int rollouts_per_mc_eval = 50;
        //     int num_repeats = 2;
        //     int num_threads = 1;
        //     int eval_threads = 1;

        //     unordered_map<string,double> alg_params =
        //     {
        //         {CZT_BIAS_PARAM_ID, 4.0},
        //         {CZT_BALL_SPLIT_VISIT_THRESH_PARAM_ID, 10.0},
        //         {SM_L_INF_THRESH_PARAM_ID, 0.05},
        //         // {SM_MAX_DEPTH, 10.0},
        //         // {SM_SPLIT_VISIT_THRESH_PARAM_ID, 10.0},
        //         {SM_SPLIT_VISIT_THRESH_PARAM_ID, 1.0},
        //         {SMBTS_SEARCH_TEMP_PARAM_ID, 100.0},
        //         {SMBTS_EPSILON_PARAM_ID, 0.1},
        //         // {SMBTS_SEARCH_TEMP_USE_DECAY_PARAM_ID, 1.0},
        //         {SMBTS_SEARCH_TEMP_USE_DECAY_PARAM_ID, 0.0},
        //         {SMBTS_SEARCH_TEMP_DECAY_VISITS_SCALE_PARAM_ID, 1.0},
        //         {SMDENTS_ENTROPY_TEMP_INIT_PARAM_ID, 0.5},
        //         {SMDENTS_ENTROPY_TEMP_VISITS_SCALE_PARAM_ID, 1.0},
        //     };

        //     vector<string> alg_ids = 
        //     {
        //         SMBTS_ALG_ID,
        //         // SMDENTS_ALG_ID,
        //         // CZT_ALG_ID,
        //         // CHMCTS_ALG_ID,
        //     };

        //     for (string alg_id : alg_ids) {
        //         run_ids->push_back(RunID(
        //             env_id,
        //             expr_id,
        //             expr_timestamp,
        //             alg_id,
        //             alg_params,
        //             search_runtime,
        //             max_trial_length,
        //             eval_delta,
        //             rollouts_per_mc_eval,
        //             num_repeats,
        //             num_threads,
        //             eval_threads
        //         ));
        //     }

        //     return run_ids;
        // }

        // // expr_id: 700_dst
        // // deep sea treasure eval
        // if (expr_id == EVAL_DST_EXPR_ID) {
        //     string env_id = DST_ENV_ID;
        //     time_t expr_timestamp = std::time(nullptr);
        //     double search_runtime = 45.0;
        //     int max_trial_length = 50;
        //     double eval_delta = 0.5;
        //     int rollouts_per_mc_eval = 2500;
        //     int num_repeats = 25;
        //     int num_threads = 16;
        //     int eval_threads = 16;

        //     // CZT
        //     unordered_map<string,double> czt_alg_params =
        //     {
        //         {CZT_BIAS_PARAM_ID, 3.96968},
        //         {CZT_BALL_SPLIT_VISIT_THRESH_PARAM_ID, 5.0},
        //     };
        //     run_ids->push_back(RunID(
        //         env_id,
        //         expr_id,
        //         expr_timestamp,
        //         CZT_ALG_ID,
        //         czt_alg_params,
        //         search_runtime,
        //         max_trial_length,
        //         eval_delta,
        //         rollouts_per_mc_eval,
        //         num_repeats,
        //         num_threads,
        //         eval_threads
        //     ));

        //     // CHMCTS
        //     unordered_map<string,double> chmcts_alg_params =
        //     {
        //         {CZT_BIAS_PARAM_ID, 33.7555},
        //         {CZT_BALL_SPLIT_VISIT_THRESH_PARAM_ID, 5.0},
        //     };
        //     run_ids->push_back(RunID(
        //         env_id,
        //         expr_id,
        //         expr_timestamp,
        //         CHMCTS_ALG_ID,
        //         chmcts_alg_params,
        //         search_runtime,
        //         max_trial_length,
        //         eval_delta,
        //         rollouts_per_mc_eval,
        //         num_repeats,
        //         num_threads,
        //         eval_threads
        //     ));

        //     // SMBTS
        //     unordered_map<string,double> smbts_alg_params =
        //     {
        //         {SM_L_INF_THRESH_PARAM_ID, 0.000144821},
        //         {SM_SPLIT_VISIT_THRESH_PARAM_ID, 1.0},
        //         {SMBTS_SEARCH_TEMP_PARAM_ID, 58.5697},
        //         {SMBTS_EPSILON_PARAM_ID, 0.414515},
        //         {SMBTS_SEARCH_TEMP_USE_DECAY_PARAM_ID, 1.0},
        //         {SMBTS_SEARCH_TEMP_DECAY_VISITS_SCALE_PARAM_ID, 99.9972},
        //     };
        //     run_ids->push_back(RunID(
        //         env_id,
        //         expr_id,
        //         expr_timestamp,
        //         SMBTS_ALG_ID,
        //         smbts_alg_params,
        //         search_runtime,
        //         max_trial_length,
        //         eval_delta,
        //         rollouts_per_mc_eval,
        //         num_repeats,
        //         num_threads,
        //         eval_threads
        //     ));

        //     // SMDENTS
        //     unordered_map<string,double> smdents_alg_params =
        //     {
        //         {SM_L_INF_THRESH_PARAM_ID, 0.000145879},
        //         {SM_SPLIT_VISIT_THRESH_PARAM_ID, 1.0},
        //         {SMBTS_SEARCH_TEMP_PARAM_ID, 64.4872},
        //         {SMBTS_EPSILON_PARAM_ID, 0.409571},
        //         {SMBTS_SEARCH_TEMP_USE_DECAY_PARAM_ID, 1.0},
        //         {SMBTS_SEARCH_TEMP_DECAY_VISITS_SCALE_PARAM_ID, 99.9981},
        //         {SMDENTS_ENTROPY_TEMP_INIT_PARAM_ID, 99.998},
        //         {SMDENTS_ENTROPY_TEMP_VISITS_SCALE_PARAM_ID, 0.0196627},
        //     };
        //     run_ids->push_back(RunID(
        //         env_id,
        //         expr_id,
        //         expr_timestamp,
        //         SMDENTS_ALG_ID,
        //         smdents_alg_params,
        //         search_runtime,
        //         max_trial_length,
        //         eval_delta,
        //         rollouts_per_mc_eval,
        //         num_repeats,
        //         num_threads,
        //         eval_threads
        //     ));

        //     // return 
        //     return run_ids;
        // }

        stringstream ss;
        ss << "Error in get_run_ids_from_expr_id for expr_id = " << expr_id;
        throw runtime_error(ss.str());
    }

    /**
     * Hyperparam optimiser - constructor
     */
    HyperparamOptimiser::HyperparamOptimiser(
        string env_id,
        string expr_id,
        time_t expr_timestamp,
        string alg_id,
        unordered_map<string, pair<double,double>> alg_params_min_max,
        double search_runtime,
        int max_trial_length,
        double eval_delta,
        int rollouts_per_mc_eval,
        int num_repeats,
        int num_threads,
        int eval_threads,
        bayesopt::Parameters params,
        ofstream &results_fs) :
            bayesopt::ContinuousModel(RELEVANT_PARAM_IDS.at(alg_id).size(), params),
            num_hyperparams(RELEVANT_PARAM_IDS.at(alg_id).size()),
            env_id(env_id),
            expr_id(expr_id),
            expr_timestamp(expr_timestamp),
            alg_id(alg_id),
            alg_param_ids(RELEVANT_PARAM_IDS.at(alg_id)),
            alg_params_min_max(alg_params_min_max),
            search_runtime(search_runtime),
            max_trial_length(max_trial_length),
            eval_delta(eval_delta),
            rollouts_per_mc_eval(rollouts_per_mc_eval),
            num_repeats(num_repeats),
            num_threads(num_threads),
            eval_threads(eval_threads),
            num_envs((eval_threads > num_threads) ? eval_threads : num_threads),
            best_eval(numeric_limits<double>::lowest()),
            best_alg_params(),
            results_fs(results_fs),
            hp_opt_iter(0)
    {
        // error checking
        if (alg_param_ids.size() != alg_params_min_max.size()) {
            throw runtime_error("Expecting list of param min/max values to be same size as list of params for alg");
        }
        for (string param_id : alg_param_ids) {
            if (!alg_params_min_max.contains(param_id)) {
                stringstream ss;
                ss << "Expected list of hyperparams for alg_id=" << alg_id 
                    << " did not match keys provided in alg_params_min_max. Specifically the param_id=" << param_id 
                    << " was missing.";
                throw runtime_error(ss.str());
            }
        }

        // might as well set bounding box here
        bayesopt::vectord min_vec(num_hyperparams);
        bayesopt::vectord max_vec(num_hyperparams);
        for (size_t i=0; i<alg_param_ids.size(); i++) {
            pair<double,double> min_max = alg_params_min_max[alg_param_ids[i]];
            min_vec[i] = min_max.first;
            max_vec[i] = min_max.second;
        }
        bayesopt::ContinuousModel::setBoundingBox(min_vec,max_vec);
    };

    bool HyperparamOptimiser::is_python_env() 
    {
        return thts::is_python_env(env_id);
    }

    unordered_map<string, double> HyperparamOptimiser::get_alg_params_from_bayesopt_vec(bayesopt::vectord vec)
    {
        unordered_map<string, double> alg_params;
        for (size_t i=0; i<alg_param_ids.size(); i++) {
            string param_id = alg_param_ids[i];
            if (BOOLEAN_PARAM_IDS.contains(param_id)) {
                pair<double,double> min_max = alg_params_min_max[param_id]; 
                alg_params[param_id] = get_bool_val_from_cts_sample(vec[i], min_max.first, min_max.second);
            } else if (INTEGER_PARAM_IDS.contains(param_id)) {
                pair<double,double> min_max = alg_params_min_max[param_id]; 
                alg_params[param_id] = get_int_val_from_cts_sample(vec[i], min_max.first, min_max.second);
            } else {
                alg_params[param_id] = vec[i];
            }
        }
        return alg_params;
    };

    bool HyperparamOptimiser::get_bool_val_from_cts_sample(double sample_val, int min, int max)
    {
        double midpoint = ((double) min+max) / 2.0;
        return (sample_val > midpoint);
    };

    int HyperparamOptimiser::get_int_val_from_cts_sample(double sample_val, int min, int max)
    {
        if (sample_val == max) {
            return max-1;            
        }
        return (int)sample_val;
    };

    /**
     * Hyperparam optimiser - fn to optimise
     */
    double HyperparamOptimiser::evaluateSample(const bayesopt::vectord &query) 
    {
        unordered_map<string,double> alg_params = get_alg_params_from_bayesopt_vec(query);
        RunID run_id(
            env_id,
            expr_id,
            expr_timestamp,
            alg_id,
            alg_params,
            search_runtime,
            max_trial_length,
            eval_delta,
            rollouts_per_mc_eval,
            num_repeats,
            num_threads,
            eval_threads
        );
        double eval = thts::run_expr(run_id, false);
        if (eval > best_eval) {
            best_eval = eval;
            best_alg_params = alg_params;
        }
        write_eval_line(alg_params, eval);
        // bayes opt tried to minimise, so return *-1.0 because want to maximise
        return -1.0 * eval;
    };

    /**
     * Writes a header with the params for each eval top results_fs
     */
    void HyperparamOptimiser::write_header()
    {
        // expr params
        results_fs 
            << "env_id,alg_id,search_runtime,max_trial_length,rollouts_per_mc_eval,num_repeats,num_threads" << endl;
        
        results_fs 
            << env_id << ","
            << alg_id << ","
            << search_runtime << ","
            << max_trial_length << ","
            << rollouts_per_mc_eval << ","
            << num_repeats << ","
            << num_threads 
            << endl << endl;
        
        // hyperparams (with sample number (hp_opt_iter) and eval at start/end)
        results_fs << "hp_opt_iter,";
        for (string& param_id : alg_param_ids) {
            results_fs << param_id << ",";
        } 
        results_fs << "eval(mc_estimate_expected_utility),best_eval_so_far" << endl;

        // Print out the min an max params trying
        results_fs << "MIN,";
        for (string param_id : alg_param_ids) {
            results_fs << alg_params_min_max[param_id].first << ",";
        } 
        results_fs << "MIN" << endl;
        results_fs << "MAX,";
        for (string param_id : alg_param_ids) {
            results_fs << alg_params_min_max[param_id].second << ",";
        } 
        results_fs << "MAX" << endl;

    };

    /**
     * Write eval/hyperparam sample line to file
     * - note that hp_opt_iter only used here, and also updated here
     */
    void HyperparamOptimiser::write_eval_line(unordered_map<string,double> alg_params, double eval)
    {   
        results_fs << hp_opt_iter++ << ",";
        for (string param_id : alg_param_ids) {
            results_fs << alg_params[param_id] << ",";
        }
        results_fs << eval << "," << best_eval << endl;
    };

    void HyperparamOptimiser::write_best_eval()
    {
        results_fs << endl;
        results_fs << "Best eval with params:" << endl;
        results_fs << "eval (mc_estimate_expected_utility) = " << best_eval << endl;
        for (pair<string,double> pr : best_alg_params) {
            results_fs << pr.first << " = " << pr.second << endl;
        }
    };

    /**
     * Gets hyperparam optimiser from expr_id
     */
    shared_ptr<HyperparamOptimiser> get_hyperparam_optimiser_from_expr_id(
        string expr_id, time_t expr_timestamp, ofstream &hp_opt_fs)
    {
        // TODO: add hp opt configs, example commented out below

        // // expr_id: 3x0 + 4x0 + 5x0
        // // initial mo gym hyperparam opt for czt
        // if (HP_OPT_MOGYM_CZT_EXPR_ID_TO_ENV_ID.contains(expr_id)) {
        //     string alg_id = CZT_ALG_ID;
        //     unordered_map<string, pair<double,double>> alg_params_min_max = {
        //         {CZT_BIAS_PARAM_ID, make_pair(0.01, 100.0)},
        //         {CZT_BALL_SPLIT_VISIT_THRESH_PARAM_ID, make_pair(1.0, 100.0)},
        //     };

        //     string env_id = HP_OPT_MOGYM_CZT_EXPR_ID_TO_ENV_ID.at(expr_id);
        //     double search_runtime = 20.0;
        //     int max_trial_length = ENV_ID_MAX_TRIAL_LEN.at(env_id);
        //     double eval_delta = 5.0;
        //     int rollouts_per_mc_eval = 1024;
        //     int num_repeats = 5;
        //     int num_threads = 16;
        //     int eval_threads = 16;

        //     bayesopt::Parameters bo_params;
        //     bo_params.surr_name = "sGaussianProcessML";
        //     bo_params.noise = 1.0; 
        //     bo_params.n_iterations = 190;
        //     bo_params.n_init_samples = 10;
        //     bo_params.n_iter_relearn = 10;
        //     bo_params.verbose_level = 0;

        //     return make_shared<HyperparamOptimiser>(
        //         env_id,
        //         expr_id,
        //         expr_timestamp,
        //         alg_id,
        //         alg_params_min_max,
        //         search_runtime,
        //         max_trial_length,
        //         eval_delta,
        //         rollouts_per_mc_eval,
        //         num_repeats,
        //         num_threads,
        //         eval_threads,
        //         bo_params,
        //         hp_opt_fs
        //     );
        // }

        stringstream ss;
        ss << "Error in get_hyperparam_optimiser_from_expr_id for expr_id = " << expr_id;
        throw runtime_error(ss.str());
    };

    /**
     * Lookup expr_id from prefix
     */
    string lookup_expr_id_from_prefix(string expr_id_prefix) 
    {
        for (const string& expr_id : ALL_EXPR_IDS) {
            if (expr_id.starts_with(expr_id_prefix)) {
                return expr_id;
            }
        }
        throw runtime_error(
            "Error looking up expr_id from prefix. Either forgot to add expr_id to 'ALL_EXPR_IDS' list or typo?");
    }
    
    /**
     * Helper to make a string of:
     * "param1=val1/param2=val2/.../paramN=valN/"
     * Old version output:
     * "param1=val1,param2=val2,...,paramN=valN",
     * but lead to filenames that were too long
    */
    string get_params_string_helper(RunID& run_id) {
        stringstream ss;
        const vector<string> &relevant_param_ids = RELEVANT_PARAM_IDS.at(run_id.alg_id);
        unordered_set<string> relevant_param_ids_set(relevant_param_ids.begin(),relevant_param_ids.end());
        for (pair<string,double> param_val_entry : run_id.alg_params) {
            if (!relevant_param_ids_set.contains(param_val_entry.first)) {
                continue;
            }
            ss << param_val_entry.first << "=" << param_val_entry.second << "/";
        }
        return ss.str();
    }

    /**
     * Gets the results directory for this run (doesn't check/make)
    */
    string get_results_dir(RunID& run_id) {
        stringstream ss;
        ss << "results/" 
            << run_id.expr_id << "_" << run_id.expr_timestamp << "/" 
            << run_id.env_id << "/" 
            << run_id.alg_id << "/"
            << get_params_string_helper(run_id);
        return ss.str();
    }
    
    /**
     * Checks if env corresponding to 'env_id' is a python env
     */
    bool is_python_env(string env_id) 
    {
        return (PY_ENVS.contains(env_id) 
            || GYM_ENVS.contains(env_id));
    }

    /**
     * Create and return the env
    */
    shared_ptr<MoThtsEnv> get_env(RunID& run_id) 
    {
        string thts_unique_filename = get_results_dir(run_id);
        string& env_id = run_id.env_id;

        if (GYM_ENVS.contains(env_id)) {
            shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
            return make_shared<GymMultiprocessingThtsEnv>(pickle_wrapper, thts_unique_filename, env_id);
        }

        // TODO: make env, frozen lake, delete below, keep the commented out python one though

        // if (env_id == FRUIT_TREE_7_ENV_ID || 
        //     env_id == FRUIT_TREE_STOCH_5_ENV_ID || 
        //     env_id == FRUIT_TREE_STOCH_7_ENV_ID)
        // {
        //     py::gil_scoped_acquire acquire;
        //     string module_name = "custom_fruit_tree";
        //     string class_name = "StochFruitTreeThtsEnv";
        //     py::dict kw_args;
        //     kw_args["depth"] = to_string((env_id == FRUIT_TREE_STOCH_5_ENV_ID) ? 5 : 7);
        //     kw_args["action_noise"] = to_string((env_id == FRUIT_TREE_7_ENV_ID) ? 0.0 : 0.2);

        //     shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
        //     return make_shared<MoPyMultiprocessingThtsEnv>(
        //         pickle_wrapper, thts_unique_filename, module_name, class_name, make_shared<py::dict>(kw_args));
        // }

        // if (env_id == IMPROVED_DST_ENV_ID ||
        //     env_id == IMPROVED_STOCH_DST_ENV_ID ||
        //     env_id == VAMPLEW_DST_ENV_ID ||
        //     env_id == VAMPLEW_STOCH_DST_ENV_ID)
        // {
        //     py::gil_scoped_acquire acquire;
        //     string module_name = "custom_deep_sea_treasure";
        //     string class_name = "ImprovedDeepSeaTreasureThtsEnv";
        //     bool is_stoch = (env_id == IMPROVED_STOCH_DST_ENV_ID || env_id == VAMPLEW_STOCH_DST_ENV_ID);
        //     bool is_vamplew = (env_id == VAMPLEW_DST_ENV_ID || env_id == VAMPLEW_STOCH_DST_ENV_ID);
        //     py::dict kw_args;
        //     kw_args["swept_by_current_prob"] = to_string(is_stoch ? 0.2 : 0.0);
        //     kw_args["is_vamplew"] = to_string(is_vamplew);

        //     shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
        //     return make_shared<MoPyMultiprocessingThtsEnv>(
        //         pickle_wrapper, thts_unique_filename, module_name, class_name, make_shared<py::dict>(kw_args));
            
        // }

        // if (DEBUG_ENVS.contains(env_id)) {
        //     int walk_len = 10;
        //     bool stochastic = (env_id == DEBUG_ENV_2_ID) || (env_id == DEBUG_ENV_4_ID);
        //     double wrong_dir_prob = stochastic ? 0.25 : 0.0;
        //     bool add_extra_rewards = (env_id == DEBUG_ENV_3_ID) || (env_id == DEBUG_ENV_4_ID);
        //     return make_shared<TestMoThtsEnv>(walk_len, wrong_dir_prob, add_extra_rewards);
        // }

        // if (DEBUG_PY_ENVS.contains(env_id)) {
        //     int walk_len = 10;
        //     bool stochastic = (env_id == DEBUG_PY_ENV_2_ID) || (env_id == DEBUG_PY_ENV_4_ID);
        //     double wrong_dir_prob = stochastic ? 0.25 : 0.0;
        //     bool add_extra_rewards = (env_id == DEBUG_PY_ENV_3_ID) || (env_id == DEBUG_PY_ENV_4_ID);

        //     py::gil_scoped_acquire acquire;
        //     string module_name = "mo_test_env";
        //     string class_name = "MoPyTestThtsEnv";
        //     py::dict kw_args;
        //     kw_args["walk_len"] = to_string(walk_len); 
        //     kw_args["wrong_dir_prob"] = to_string(wrong_dir_prob);
        //     kw_args["add_extra_rewards"] = to_string(add_extra_rewards);

        //     shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
        //     return make_shared<MoPyMultiprocessingThtsEnv>( 
        //         pickle_wrapper, thts_unique_filename, module_name, class_name, make_shared<py::dict>(kw_args));
        // }

        stringstream ss;
        ss << "Error in get_env for env_id = " << env_id;
        throw runtime_error(ss.str());
    }

}