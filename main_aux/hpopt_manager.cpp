#include "main_aux/hpopt_manager.h"

#include "helper.h"

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

#include <cmath>
#include <limits>
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
    HpoptManager::HpoptManager(std::time_t xpr_timestamp, HpoptConfigMap xpr_config, HpoptConfigMap alg_config) :
        bayesopt::ContinuousModel(alg_config.size()-1, params), 
        xpr_timestamp(xpr_timestamp), 
        xpr_config(xpr_config), 
        alg_config(alg_config), 
        num_hyperparameters(alg_config.size()-1),
        best_thts_manager(nullptr),
        best_thts_manager_eval(std::numeric_limits<double>::lowest()),
        best_thts_manager_std_mean_eval(std::numeric_limits<double>::lowest()),
        write_eval_logs(false),
        hpopt_summary_fs()
    {
        validate_config_or_raise_exception();
        set_bayesopt_bounding_box();
    }

    /**
     *  Checks all expected params are present, and that no additional params
     */  
    HpoptManager::validate_config_or_raise_exception()
    {
        if (xpr_config.size() != 14)
        {
            throw runtime_error("Expecting 14 entries in the xpr level config.");
        }

        if (get_config_value(xpr_config, XPR_OR_ALG_ID_TAG) != HPOPT_PARAMS_ID_TAG)
        {
            throw runtime_error("In hpopt manager expecting config entry: {XPR_OR_ALG_ID_TAG,HPOPT_PARAMS_ID_TAG}");
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
            HPOPT_PARAM_ID_MIN_REPEATS,
            HPOPT_PARAM_ID_ESTIMATE_CONFIDENCE_THRESHOLD,
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
    vector<HpoptConfigMap> HpoptManager::lookup_config_vector_from_xpr_prefix(string xpr_id_prefix)
    {
        // Validate that all configs have the first HpoptConfigMap with xpr level config, including an xpr_id
        for (vector<HpoptConfigMap>& config : all_hpopt_configs)
        {
            HpoptConfigMap& xpr_config = config[0];
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
        for (vector<HpoptConfigMap>& config : ALL_CONFIGS)
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
     * Config -> HpoptManagers
     */
    shared_ptr<vector<HpoptManager>> HpoptManager::get_hpopt_managers_from_config_vector(
        vector<HpoptConfigMap>& config_vector)
    {
        time_t xpr_timestamp = std::time(nullptr);
        shared_ptr<vector<HpoptManager>> hpopt_managers = std::make_shared<vector<HpoptManager>>();
        for (size_t i=1; i<config_vector.size(); i++)
        {  
            hpopt_managers->push_back(HpoptManager(xpr_timestamp, config_vector[0], config_vector[i]));
        }
        return hpopt_managers;
    }

    /**
     * Sets the bounding box for bayesopt to sample from
     * Sets it to a uniform hypercube, we'll do the work getting it into ranges we want ourselves, as want custom log 
     * sampling stuff
     */
    void set_bayesopt_bounding_box()
    {
        bayesopt::vectord min_vec(num_hyperparams);
        bayesopt::vectord max_vec(num_hyperparams);
        for (size_t i=0; i<num_hyperparams; i++) 
        {
            min_vec[i] = 0.0;
            max_vec[i] = 1.0;
        }
        bayesopt::ContinuousModel::setBoundingBox(min_vec,max_vec);
    }

    /**
     * Helper function to compute mean and std of vector of evals
     * 
     * See evaluateSample docstring, it's the updates for the statistics
     * 
     * Returns (via references):
     *  :mean: mean of the evals
     *  :std: std as above
     *  :std_mean_eval: the value of sqrt(Var(VBar)) = std/sqrt(n)
     */
    void update_statistics(const vector<double>& evals, double& mean_eval, double& std_eval, double& std_mean_eval)
    {
        if (evals.size() < 2)
        {
            mean_eval = 0.0;
            std_eval = std::numeric_limits<double>::max();
            std_mean_eval = std::numeric_limits<double>::max();
        }

        double evals_sum = 0.0;
        for (double eval : evals) 
        {
            evals_sum += eval;
        }
        mean_eval = evals_sum / evals.size();

        double std_eval_sum = 0.0;
        for (double eval : evals) 
        {
            std_eval_sum += (eval - mean_eval) * (eval - mean_eval);
        }
        std_eval = sqrt(std_eval_sum / (evals.size() - 1));
        std_mean_eval = std_eval / sqrt(evals.size());
    }

    /**
     * Bayesopt intefrace.
     * 
     * Will repeatedly evaluate some parameters until both of the conditions hold:
     *  num_repeats > min_repeats and
     *  estimate_confidence_threshold > sqrt(Var(mean_eval))
     * 
     * If Vbar is the mean estimate, V1 is a rv for value esimate of a run, and std^2=Var(V1)
     * Then after n repeats, Var(Vbar) = Var(V1) / n approx= std^2 / n
     * So when std^2 / n < mean_estimate_variance_threshold, then we can stop
     * 
     * As std of the mean_eval is what we care about, we keep track of three statistics:
     *  mean_eval = mean eval over all evaluations/repeats
     *  std_eval = std of evals over all evaluations/repeats
     *  std_mean_eval = estimate of std of mean_eval using all evaluations/repeats
     * 
     * Returns the NEGATIVE of the mean value estimate at the root node from the repeated runs
     * Bayesopt r 
     * 
     * Args:
     *  :query: a vector sampled by bayes opt, to return an evaluation score for
     * Returns:
     *  :evaluation_score: a score to be MINIMISED by bayesopt
     */
    double evaluateSample(const bayesopt::vectord& query) override
    {
        int repeats_run = 0;
        vector<double> evals;
        double mean_eval = 0.0;
        double std_eval = 0.0;
        double std_mean_eval = 0.0;

        int min_repeats = get_hpopt_min_repeats();
        double estimate_confidence_threshold == get_hpopt_estimate_confidence_threshold();
        
        ofstream eval_log_fs;
        if (write_eval_logs)
        {
            eval_log_fs = get_eval_log_filestream(manager, repeats_run-1);
            write_eval_log_header(eval_log_fs, manager);
        }   

        // run evals
        while (repeats_run < min_repeats || std_mean_eval > estimate_confidence_threshold)
        {
            shared_ptr<ThtsEnv> env = get_env();
            shared_ptr<ThtsManager> manager = get_thts_manager(env, query);
            shared_ptr<ThtsDNode> root_node = get_root_search_node(env, manager);

            double eval = thts::run_eval(manager, root_node, is_python_env());

            evals.push_back(eval);
            compute_mean_and_std_(evals, mean_eval, std_eval, std_mean_eval);
            repeats_run++;

            cout << "Hp_opt_iter " << hp_opt_iter 
                << ". mean_eval=" << mean_eval 
                << ",std_mean_eval=" << std_mean_eval << " >? " << std_mean_eval_threshold << endl;
                   
            if (write_eval_logs)
            {
                write_eval_log(eval_log_fs, repeats_run-1, eval, get_termination_bound(), 0.0, get_num_eval_rollouts());
            }
        }

        // Update if best eval so far
        if (mean_eval > best_thts_manager_eval)
        {
            best_thts_manager = thts_manager;
            best_thts_manager_mean_eval = mean_eval;
            best_thts_manager_std_mean_eval = std_mean_eval;
        }
        
        // Write to logs
        write_hpopt_summary_sample_eval_line(manager, mean_eval, std_mean_eval);

        // Return sample eval
        hp_opt_iter++;
        return -1.0 * mean_eval;
    }

    /**
     * Getters - xpr level config
     */
    string HpoptManager::get_xpr_name()           { return get_config_value(xpr_config, XPR_PARAM_ID_NAME); }
    string HpoptManager::get_env_id()             { return get_config_value(xpr_config, XPR_PARAM_ID_ENV); }
    bool HpoptManager::get_mcts_mode()            { return get_config_value(xpr_config, XPR_PARAM_ID_MCTS_MODE); }
    bool HpoptManager::get_graph_search()         { return get_config_value(xpr_config, XPR_PARAM_ID_GRAPH_SEARCH); }
    int HpoptManager::get_max_trial_length()      { return get_config_value(xpr_config, XPR_PARAM_ID_MAX_TRIAL_LENGTH); }
    bool HpoptManager::xpr_is_runtime_bounded()   { return get_config_value(xpr_config, XPR_PARAM_ID_RUNTIME_BOUNDED); }
    double HpoptManager::get_termination_bound()  { return get_config_value(xpr_config, XPR_PARAM_ID_TERMINATION_BOUND); }
    int HpoptManager::get_repeated_runs_per_alg() { return get_config_value(xpr_config, XPR_PARAM_ID_REPEATED_RUNS_PER_ALG); }
    int HpoptManager::get_num_search_threads()    { return get_config_value(xpr_config, XPR_PARAM_ID_SEARCH_THREADS); }
    double HpoptManager::get_eval_delta()         { return get_config_value(xpr_config, XPR_PARAM_ID_EVAL_DELTA); }
    int HpoptManager::get_num_eval_rollouts()     { return get_config_value(xpr_config, XPR_PARAM_ID_EVAL_ROLLOUTS); }
    int HpoptManager::get_num_eval_threads()      { return get_config_value(xpr_config, XPR_PARAM_ID_EVAL_THREADS); }

    /**
     * Getters - hpopt (xpr) level config
     */
    int HpoptManager::get_hpopt_min_repeats()                       { return get_config_value(xpr_config, HPOPT_PARAM_ID_MIN_REPEATS); }
    double HpoptManager::get_hpopt_estimate_confidence_threshold()  { return get_config_value(xpr_config, HPOPT_PARAM_ID_ESTIMATE_CONFIDENCE_THRESHOLD); }

    /**
     * Getters - alg level config
     */
    string HpoptManager::get_alg_id()                                   { return get_config_value(alg_config, XPR_OR_ALG_ID_TAG); }
    pair<double,double> HpoptManager::get_bias_range()                  { return get_config_value(alg_config, ALG_PARAM_ID_BIAS); }
    pair<int,int> HpoptManager::get_uct_budget_range()                  { return get_config_value(alg_config, ALG_PARAM_ID_UCT_BUDGET); }
    pair<double,double> HpoptManager::get_init_temp_range()             { return get_config_value(alg_config, ALG_PARAM_ID_INIT_TEMP); }
    pair<double,double> HpoptManager::get_temp_decay_rate_range()       { return get_config_value(alg_config, ALG_PARAM_ID_TEMP_DECAY_RATE); }
    pair<double,double> HpoptManager::get_init_entropy_coeff_range()    { return get_config_value(alg_config, ALG_PARAM_ID_INIT_ENTROPY_COEFF); }
    pair<double,double> HpoptManager::get_entropy_zero_at_range()       { return get_config_value(alg_config, ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT); }
    pair<double,double> HpoptManager::get_epsilon_range()               { return get_config_value(alg_config, ALG_PARAM_ID_EPSILON); }
    pair<double,double> HpoptManager::get_default_q_value_range()       { return get_config_value(alg_config, ALG_PARAM_ID_DEFAULT_Q_VALUE); }

            
    /**
     * Helper to sample boolean value using a continuous [0,1] random variable from bayesopt
     */
    bool HpoptManager::get_bool_val_from_bayesopt_sample(double sample_val)
    {
        return (sample_val > 0.5);
    }

    /**
     * Helper to sample integer value using a continuous [0,1] random variable from bayesopt
     */
    int HpoptManager::get_int_val_from_bayesopt_sample(double sample_val, int min, int max)
    {
        if (sample_val == max) {
            return max-1;            
        }
        return (int)sample_val;
    }

    /**
     * Helper to sample double value using a continuous [0,1] random variable from bayesopt
     * (N.B. we may want to apply log scaling)
     */
    double HpoptManager::get_cts_val_from_bayesopt_sample(double sample_val, int min, int max, bool log_scaling)
    {
        if (log_scaling)
        {
            min = log(min);
            max = log(max);
        }

        double denormalized_sample = (1.0-sample_val) * min + sample_val * max;

        if (log_scaling)
        {
            denormalized_sample = exp(denormalized_sample);
        }

        return denormalized_sample;
    }

    /**
     * Samplers - alg level config - returns sampled values using [0,1] uniform random sample from bayesopt
     */
    double HpoptManager::sample_bias(double rand)
    {
        pair<double,double> min_max = get_bias_range();
        double min = min_max.first;
        double max = min_max.second;
        return get_cts_val_from_bayesopt_sample(rand, min, max, HPOPT_LOG_SCALE_ALG_PARAM_IDS.contains(ALG_PARAM_ID_BIAS));
    }

    int HpoptManager::sample_uct_budget(double rand)
    {
        pair<int,int> min_max = get_uct_budget_range();
        double min = min_max.first;
        double max = min_max.second;
        return get_int_val_from_bayesopt_sample(rand, min, max);
    }
    
    double HpoptManager::sample_init_temp(double rand)
    {
        pair<double,double> min_max = get_init_temp_range();
        double min = min_max.first;
        double max = min_max.second;
        return get_cts_val_from_bayesopt_sample(rand, min, max, HPOPT_LOG_SCALE_ALG_PARAM_IDS.contains(ALG_PARAM_ID_INIT_TEMP));
    }
    
    double HpoptManager::sample_temp_decay_rate(double rand)
    {
        pair<double,double> min_max = get_temp_decay_rate_range();
        double min = min_max.first;
        double max = min_max.second;
        return get_cts_val_from_bayesopt_sample(rand, min, max, HPOPT_LOG_SCALE_ALG_PARAM_IDS.contains(ALG_PARAM_ID_TEMP_DECAY_RATE));
    }
    
    double HpoptManager::sample_init_entropy_coeff(double rand)
    {
        pair<double,double> min_max = get_init_entropy_coeff_range();
        double min = min_max.first;
        double max = min_max.second;
        return get_cts_val_from_bayesopt_sample(rand, min, max, HPOPT_LOG_SCALE_ALG_PARAM_IDS.contains(ALG_PARAM_ID_INIT_ENTROPY_COEFF));
    }
    
    double HpoptManager::sample_entropy_zero_at(double rand)
    {
        pair<double,double> min_max = get_entropy_zero_at_range();
        double min = min_max.first;
        double max = min_max.second;
        return get_cts_val_from_bayesopt_sample(rand, min, max, HPOPT_LOG_SCALE_ALG_PARAM_IDS.contains(ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT));
    }
    
    double HpoptManager::sample_epsilon(double rand)
    {
        pair<double,double> min_max = get_epsilon_range();
        double min = min_max.first;
        double max = min_max.second;
        return get_cts_val_from_bayesopt_sample(rand, min, max, HPOPT_LOG_SCALE_ALG_PARAM_IDS.contains(ALG_PARAM_ID_EPSILON));
    }
    
    double HpoptManager::sample_default_q_value(double rand)
    {
        pair<double,double> min_max = get_default_q_value_range();
        double min = min_max.first;
        double max = min_max.second;
        return get_cts_val_from_bayesopt_sample(rand, min, max, HPOPT_LOG_SCALE_ALG_PARAM_IDS.contains(ALG_PARAM_ID_DEFAULT_Q_VALUE));
    }
    

    /**
     * Returns if the env we are using is a python env
    */
    bool HpoptManager::is_python_env()
    {
        return (PY_ENVS.contains(get_env_id()) || GYM_ENVS.contains(get_env_id()));
    }

    /**
     * Returns an instance of ThtsEnv to use for this run
    */
    shared_ptr<ThtsEnv> HpoptManager::get_env()
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
    shared_ptr<ThtsManager> HpoptManager::get_thts_manager(shared_ptr<ThtsEnv> env, const bayesopt::vectord& query)
    {
        string alg_id = get_alg_id();
        shared_ptr<ThtsManager> thts_manager = nullptr;

        if (alg_id == ALG_ID_UCT || alg_id == ALG_ID_MAX_UCT) 
        {
            UctManagerArgs manager_args(env);
            manager_args.bias = sample_bias(query[0]);
            thts_manager = make_shared<UctManager>(manager_args);
        }

        else if (alg_id == ALG_ID_HMCTS)
        {
            HmctsManagerArgs manager_args(env);
            manager_args.bias = sample_bias(query[0]);
            manager_args.uct_budget_threshold = sample_uct_budget(query[1]);
            thts_manager = make_shared<HmctsManager>(manager_args);
        }

        else if (alg_id == ALG_ID_MENTS || alg_id == ALG_ID_RENTS || alg_id == ALG_ID_TENTS)
        {
            MentsManagerArgs manager_args(env);
            manager_args.temp_schedule_ptr = make_shared<SqrtSchedule>(
                sample_init_temp(query[0]), sample_temp_decay_rate(query[1]));
            manager_args.epsilon = sample_epsilon(query[2]);
            manager_args.default_q_value = sample_default_q_value(query[3]);
            thts_manager = make_shared<MentsManager>(manager_args);
        }

        else if (alg_id == ALG_ID_BTS || alg_id == ALG_ID_DENTS)
        {
            DentsManagerArgs manager_args(env);
            manager_args.temp_schedule_ptr = make_shared<SqrtSchedule>(
                sample_init_temp(query[0]), sample_temp_decay_rate(query[1]));
            manager_args.epsilon = sample_epsilon(query[2]);
            manager_args.default_q_value = sample_default_q_value(query[3]);
            
            if (alg_id == ALG_ID_DENTS)
            {
                manager_args.entropy_coeff_schedule_ptr = make_shared<LinearSchedule>(
                    sample_init_entropy_coeff(query[4]), sample_entropy_zero_at(query[5]));
            }

            thts_manager = make_shared<DentsManager>(manager_args);
        }
        
        if (thts_manager == nullptr)
        {
            stringstream ss;
            ss << "Error in HpoptManager get_thts_manager for alg_id = " << alg_id;
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
    shared_ptr<ThtsDNode> HpoptManager::get_root_search_node(shared_ptr<ThtsEnv> env, shared_ptr<ThtsManager> manager)
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
     * Helper to get a string of values from thts_manager, seperated by commas
     */
    ConfigMap HpoptManager::config_map_from_thts_manager(stdshared_ptr<ThtsManager> manager)
    {
        ConfigMap instance_config_map;

        string alg_id = get_alg_id();

        if (alg_id == ALG_ID_UCT || alg_id == ALG_ID_MAX_UCT) 
        {
            shared_ptr<UctManager> mngr = static_pointer_cast<UctManager>(manager);
            instance_config_map[ALG_PARAM_ID_BIAS] = mngr->bias;
        }

        else if (alg_id == ALG_ID_HMCTS)
        {
            shared_ptr<HmctsManager> mngr = static_pointer_cast<HmctsManager>(manager);
            instance_config_map[ALG_PARAM_ID_BIAS] = mngr->bias;
            instance_config_map[ALG_PARAM_ID_UCT_BUDGET] = mngr->uct_budget;
        }

        else if (alg_id == ALG_ID_MENTS || alg_id == ALG_ID_RENTS || alg_id == ALG_ID_TENTS)
        {
            shared_ptr<MentsManager> mngr = static_pointer_cast<MentsManager>(manager);
            SqrtSchedule& temp_schedule = *mngr->temp_schedule_ptr;
            instance_config_map[ALG_PARAM_ID_INIT_TEMP] = temp_schedule.get_temp_at_zero_visits();
            instance_config_map[ALG_PARAM_ID_TEMP_DECAY_RATE] = temp_schedule.get_decay_rate_coeff();
            instance_config_map[ALG_PARAM_ID_EPSILON] = mngr->epsilon;
            instance_config_map[ALG_PARAM_ID_DEFAULT_Q_VALUE] = mngr->default_q_value;
        }

        else if (alg_id == ALG_ID_BTS || alg_id == ALG_ID_DENTS)
        {

            shared_ptr<DentsManager> mngr = static_pointer_cast<DentsManager>(manager);
            SqrtSchedule& temp_schedule = *mngr->temp_schedule_ptr;
            instance_config_map[ALG_PARAM_ID_INIT_TEMP] = temp_schedule.get_temp_at_zero_visits();
            instance_config_map[ALG_PARAM_ID_TEMP_DECAY_RATE] = temp_schedule.get_decay_rate_coeff();
            instance_config_map[ALG_PARAM_ID_EPSILON] = mngr->epsilon;
            instance_config_map[ALG_PARAM_ID_DEFAULT_Q_VALUE] = mngr->default_q_value;
            
            if (alg_id == ALG_ID_DENTS)
            {
                LinearSchedule& ent_coeff_schedule = *mngr->entropy_coeff_schedule_ptr;
                instance_config_map[ALG_PARAM_ID_INIT_ENTROPY_COEFF] = ent_coeff_schedule.get_temp_at_zero_visits();
                instance_config_map[ALG_PARAM_ID_ENTROPY_COEFF_ZERO_AT] = ent_coeff_schedule.zero_temp_at();
            }
        }
        
        return instance_config_map;
    }

    /**
     * A unique filename for this manager to write hpopt summary to
     */
    std::filesystem::path HpoptManager::get_hpopt_summary_filename()
    {
        stringstream ss;
        ss << "aux_hpopt_summaries/" << get_xpr_name() << "_" << xpr_timestamp << "_alg_" << get_alg_id() << ".txt";
        std::filesystem::path filepath(ss.str());
        return filepath;
    }

    /**
     * Create directories if needbe
     * And return filestream to summary file
     */
    ofstream HpoptManager::get_hpopt_summary_filestream()
    {
        std::filesystem::path filepath = get_hpopt_summary_filename();
        std::filesystem::path dir = file_path.parent_path();

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

    void HpoptManager::open_hpopt_summary_filestream()
    {
        hpopt_summary_fs = get_hpopt_summary_filestream();
    }

    void HpoptManager::close_hpopt_summary_filestream()
    {
        hpopt_summary_fs.close();
    }
    
    /**
     * Functions to write the header, saying the params and ranges being searched over
     */
    void HpoptManager::write_hpopt_summary_header(ofstream& fs)
    {
        // Xpr level params
        hpopt_summary_fs << "Hpopt Xpr level params:" << endl << endl;;
        hpopt_summary_fs << XPR_PARAM_ID_NAME << ","
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
        hpopt_summary_fs << get_xpr_name() << ","
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
        hpopt_summary_fs << endl << alg_id << " params mins/maxs: " << endl << endl;;
        for (string alg_param_id : ALG_ID_TO_ALG_PARAM_IDS[alg_id])
        {
            hpopt_summary_fs << alg_param_id << " - (min,max,log_scale) = (";
            hpopt_summary_fs << get_config_value(alg_config, alg_param_id).first << ","
                << get_config_value(alg_config, alg_param_id).second << ","
                << (HPOPT_LOG_SCALE_ALG_PARAM_IDS.contains(alg_param_id)) << ")" << endl;
        }
        
        // csv header for eval lines
        hpopt_summary_fs << endl << "Evaluations:" << endl << endl;
        hpopt_summary_fs << "hp_opt_iter,mean_eval,std_mean_eval,best_eval_so_far"
        for (string alg_param_id : ALG_ID_TO_ALG_PARAM_IDS[alg_id])
        {
            hpopt_summary_fs << "," << alg_param_id;
        }
        hpopt_summary_fs << endl;
    }

    void HpoptManager::write_hpopt_summary_sample_eval_line(
        shared_ptr<ThtsManager> manager, double mean_eval, double std_mean_eval)
    {
        string alg_id = get_alg_id();
        ConfigMap alg_params = config_map_from_thts_manager(manager);

        hpopt_summary_fs << hp_opt_iter << "," << mean_eval << "," << std_mean_eval << "," << best_thts_manager_eval;
        for (string& alg_param_id : ALG_ID_TO_ALG_PARAM_IDS[alg_id])
        {
            hpopt_summary_fs << "," << alg_params[alg_param_id];
        }
        hpopt_summary_fs << endl;
    }
    
    void HpoptManager::write_hpopt_summary_footer()
    {   
        hpopt_summary_fs << endl  << "Best Params: " << endl << endl;
        hpopt_summary_fs << "mean_eval - " << best_thts_manager_mean_eval << endl;
        hpopt_summary_fs << "std_mean_eval - " << best_thts_manager_sdt_mean_eval << endl;
        throw runtime_error("not_implemented");
    }

    /**
     * Helper to make a string of:
     * "param1=val1/param2=val2/.../paramN=valN/"
     * Old version output:
     * "param1=val1,param2=val2,...,paramN=valN",
     * but lead to filenames that were too long
    */
    string get_params_string_helper(HpoptManager& run_manager) 
    {
        stringstream ss;
        const vector<string>& relevant_alg_param_ids = ALG_ID_TO_ALG_PARAM_IDS.at(run_manager.get_alg_id());
        for (const string& alg_param_id : relevant_alg_param_ids)
        {
            ss << alg_param_id << "=" << get_config_value<double>(run_manager.alg_config, alg_param_id) << "/";
        }
        return ss.str();
    }

    /**
     * Gets the results directory for this run (doesn't check/make)
    */
    string HpoptManager::get_eval_logs_dir() 
    {
        stringstream ss;
        ss << "eval_logs_aux_hpopt/" 
            << get_xpr_name() << "_" << xpr_timestamp << "/"
            << get_env_id() << "/"
            << get_alg_id() << "/"
            << get_params_string_helper(*this);
        return ss.str();
    }

    std::filesystem::path HpoptManager::get_eval_log_filename(std::shared_ptr<ThtsManager> manager, int run_idx)
    {
        stringstream filename_ss;
        filename_ss << "eval_log_run_idx_" << run_idx << ".txt";

        std::filesystem::path dir = get_eval_logs_dir();
        std::filesystem::path filename = dir / filename_ss.str();

        return filename;
    
    ofstream HpoptManager::get_eval_log_filestream(std::shared_ptr<ThtsManager> manager, int run_idx)
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

    /**
     * Functions for writing to logs files
     */
    void HpoptManager::write_eval_log_header(std::ofstream& fs, shared_ptr<ThtsManager> manager)
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
            << XPR_PARAM_ID_EVAL_THREADS << ","
            << HPOPT_PARAM_ID_MIN_REPEATS << ","
            << HPOPT_PARAM_ID_ESTIMATE_CONFIDENCE_THRESHOLD << endl;
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
            << get_num_eval_threads() << ","
            << get_hpopt_min_repeats() << ","
            << get_hpopt_estimate_confidence_threshold() << endl;

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
        results_evals_fs << "run_idx,eval,num_trials,runtime,num_eval_samples" << endl;
    }

    void HpoptManager::write_eval_log(
        ofstream& fs, int run_idx, double eval, int num_trials, double runtime, int num_eval_samples)
    {
        fs << run_idx << "," << eval << "," << num_trials << "," << runtime << "," << num_eval_samples << endl;
    }
}
