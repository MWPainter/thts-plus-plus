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

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
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
    HpoptManager::HpoptManager(std::time_t xpr_timestamp, HpoptConfigMap xpr_config, HpoptConfigMap alg_config, bayesopt::Parameters params) :
        bayesopt::ContinuousModel(alg_config.size()-1, params), 
        xpr_timestamp(xpr_timestamp), 
        xpr_config(xpr_config), 
        alg_config(alg_config), 
        num_hyperparams(alg_config.size()-1),
        best_config_map(),
        best_mean_eval(std::numeric_limits<double>::lowest()),
        best_std_mean_eval(std::numeric_limits<double>::lowest()),
        hpopt_summary_fs(),
        hp_opt_iter(0),
        bo_params(params)
    {
        validate_config_or_raise_exception();
        set_bayesopt_bounding_box();
    }

    /**
     * Copy constructor
     */
    HpoptManager::HpoptManager(const HpoptManager& other) :
        bayesopt::ContinuousModel(other.num_hyperparams, other.bo_params),
        xpr_timestamp(other.xpr_timestamp),
        xpr_config(other.xpr_config),
        alg_config(other.alg_config),
        num_hyperparams(other.alg_config.size()-1),
        best_config_map(other.best_config_map),
        best_mean_eval(other.best_mean_eval),
        best_std_mean_eval(other.best_std_mean_eval),
        hpopt_summary_fs(),
        hp_opt_iter(other.hp_opt_iter),
        bo_params(other.bo_params)
    {
        validate_config_or_raise_exception();
        set_bayesopt_bounding_box();
    }

    /**
     * Destructor
     */
    HpoptManager::~HpoptManager()
    {
        hpopt_summary_fs.close();
    }

    /**
     *  Checks all expected params are present, and that no additional params
     */  
    void HpoptManager::validate_config_or_raise_exception()
    {
        if (xpr_config.size() != 17)
        {
            throw runtime_error("Expecting 17 entries in the xpr level config.");
        }

        if (get_config_value<std::string>(xpr_config, XPR_OR_ALG_ID_TAG) != HPOPT_PARAMS_ID_TAG)
        {
            throw runtime_error("In hpopt manager expecting config entry: {XPR_OR_ALG_ID_TAG,HPOPT_PARAMS_ID_TAG}");
        }

        vector<string> xpr_param_ids = 
        {
            XPR_PARAM_ID_NAME, 
            XPR_PARAM_ID_ENV, 
            XPR_PARAM_ID_MCTS_MODE, 
            XPR_PARAM_ID_GRAPH_SEARCH,
            XPR_PARAM_ID_MAX_TRIAL_LENGTH,
            XPR_PARAM_ID_RUNTIME_BOUNDED, 
            XPR_PARAM_ID_TERMINATION_BOUND, 
            // XPR_PARAM_ID_REPEATED_RUNS_PER_ALG, 
            XPR_PARAM_ID_SEARCH_THREADS, 
            XPR_PARAM_ID_EVAL_DELTA, 
            XPR_PARAM_ID_EVAL_ROLLOUTS, 
            XPR_PARAM_ID_EVAL_THREADS,
            HPOPT_PARAM_ID_MIN_REPEATS,
            HPOPT_PARAM_ID_ESTIMATE_CONFIDENCE_THRESHOLD,
            HPOPT_PARAM_ID_BAYESOPT_TOTAL_SAMPLES,
            HPOPT_PARAM_ID_BAYESOPT_INIT_RAND_SAMPLES,
            HPOPT_PARAM_ID_BAYESOPT_RELEARN_FREQ,
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
     * Sets the bounding box for bayesopt to sample from
     * Sets it to a uniform hypercube, we'll do the work getting it into ranges we want ourselves, as want custom log 
     * sampling stuff
     */
    void HpoptManager::set_bayesopt_bounding_box()
    {
        bayesopt::vectord min_vec(num_hyperparams);
        bayesopt::vectord max_vec(num_hyperparams);
        for (int i=0; i<num_hyperparams; i++) 
        {
            min_vec[i] = 0.0;
            max_vec[i] = 1.0;
        }
        bayesopt::ContinuousModel::setBoundingBox(min_vec,max_vec);
    }

    /**
     * Lookup config from xpr_id_prefix, so unique id's, but not pain to type
     */
    vector<HpoptConfigMap> HpoptManager::lookup_config_vector_from_xpr_prefix(string xpr_id_prefix)
    {
        // Validate that all configs have the first HpoptConfigMap with xpr level config, including an xpr_id
        for (const vector<HpoptConfigMap>& config : ALL_HPOPT_CONFIGS)
        {
            const HpoptConfigMap& xpr_config = config[0];
            if (get_config_value<std::string>(xpr_config, XPR_OR_ALG_ID_TAG) != HPOPT_PARAMS_ID_TAG)
            {
                throw runtime_error("Expecting first map in each config (vector) to specify xpr level config with correct tagging.");
            }
            if (!xpr_config.contains(XPR_PARAM_ID_NAME)) 
            {
                throw runtime_error("Expecting xpr level config to specify an xpr_name");
            }
        }

        // Lookup
        for (const vector<HpoptConfigMap>& config : ALL_HPOPT_CONFIGS)
        {
            const string& xpr_name = get_config_value<std::string>(config[0], XPR_PARAM_ID_NAME);
            if (xpr_name.starts_with(xpr_id_prefix))
            {
                return vector<HpoptConfigMap>(config);
            }
        }

        stringstream ss;
        ss << "Error looking up xpr_id from prefix, couldn't find config starting with " << xpr_id_prefix << "in ALL_CONFIGS";
        throw runtime_error(ss.str());
    }

    /**
     * Config -> BayesOpt params
     */
    bayesopt::Parameters HpoptManager::get_bayesopt_params_from_xpr_config(HpoptConfigMap& xpr_config)
    {
        double target_std_per_bayesopt_sample = get_config_value<double>(xpr_config, HPOPT_PARAM_ID_ESTIMATE_CONFIDENCE_THRESHOLD);
        int bayesopt_total_samples = get_config_value<int>(xpr_config, HPOPT_PARAM_ID_BAYESOPT_TOTAL_SAMPLES);
        int bayesopt_init_rand_samples = get_config_value<int>(xpr_config, HPOPT_PARAM_ID_BAYESOPT_INIT_RAND_SAMPLES);
        int bayesopt_relearn_freq = get_config_value<int>(xpr_config, HPOPT_PARAM_ID_BAYESOPT_RELEARN_FREQ);

        bayesopt::Parameters bo_params;
        bo_params.surr_name = "sGaussianProcessML";
        // bo_params.surr_name = "sGaussianProcessNormal";
        bo_params.noise = target_std_per_bayesopt_sample*target_std_per_bayesopt_sample;
        bo_params.n_iterations = bayesopt_total_samples - bayesopt_init_rand_samples;
        bo_params.n_init_samples = bayesopt_init_rand_samples;
        bo_params.n_iter_relearn = bayesopt_relearn_freq;
        bo_params.verbose_level = 0;

        return bo_params;
    }

    /**
     * Config -> HpoptManagers
     */
    shared_ptr<vector<HpoptManager>> HpoptManager::get_hpopt_managers_from_config_vector(
        vector<HpoptConfigMap>& config_vector)
    {
        time_t xpr_timestamp = std::time(nullptr);
        bayesopt::Parameters bo_params = get_bayesopt_params_from_xpr_config(config_vector[0]);
        shared_ptr<vector<HpoptManager>> hpopt_managers = std::make_shared<vector<HpoptManager>>();
        for (size_t i=1; i<config_vector.size(); i++)
        {  
            // Use emplace_back to construct in-place, avoiding move/copy
            hpopt_managers->emplace_back(xpr_timestamp, config_vector[0], config_vector[i], bo_params);
        }
        return hpopt_managers;
    }

    /**
     * Getters - xpr level config
     */
    string HpoptManager::get_xpr_name()           { return get_config_value<std::string>(xpr_config, XPR_PARAM_ID_NAME); }
    string HpoptManager::get_env_id()             { return get_config_value<std::string>(xpr_config, XPR_PARAM_ID_ENV); }
    bool HpoptManager::get_mcts_mode()            { return get_config_value<bool>(xpr_config, XPR_PARAM_ID_MCTS_MODE); }
    bool HpoptManager::get_graph_search()         { return get_config_value<bool>(xpr_config, XPR_PARAM_ID_GRAPH_SEARCH); }
    int HpoptManager::get_max_trial_length()      { return get_config_value<int>(xpr_config, XPR_PARAM_ID_MAX_TRIAL_LENGTH); }
    bool HpoptManager::xpr_is_runtime_bounded()   { return get_config_value<bool>(xpr_config, XPR_PARAM_ID_RUNTIME_BOUNDED); }
    double HpoptManager::get_termination_bound()  { return get_config_value<double>(xpr_config, XPR_PARAM_ID_TERMINATION_BOUND); }
    int HpoptManager::get_repeated_runs_per_alg() { return get_config_value<int>(xpr_config, XPR_PARAM_ID_REPEATED_RUNS_PER_ALG); }
    int HpoptManager::get_num_search_threads()    { return get_config_value<int>(xpr_config, XPR_PARAM_ID_SEARCH_THREADS); }
    double HpoptManager::get_eval_delta()         { return get_config_value<double>(xpr_config, XPR_PARAM_ID_EVAL_DELTA); }
    int HpoptManager::get_num_eval_rollouts()     { return get_config_value<int>(xpr_config, XPR_PARAM_ID_EVAL_ROLLOUTS); }
    int HpoptManager::get_num_eval_threads()      { return get_config_value<int>(xpr_config, XPR_PARAM_ID_EVAL_THREADS); }

    /**
     * Getters - alg level config
     */
    string HpoptManager::get_alg_id()             { return get_config_value<std::string>(alg_config, XPR_OR_ALG_ID_TAG); }

    /**
     * Getters - hpopt (xpr) level config
     */
    int HpoptManager::get_hpopt_min_repeats()                       { return get_config_value<int>(xpr_config, HPOPT_PARAM_ID_MIN_REPEATS); }
    double HpoptManager::get_hpopt_estimate_confidence_threshold()  { return get_config_value<double>(xpr_config, HPOPT_PARAM_ID_ESTIMATE_CONFIDENCE_THRESHOLD); }
    int HpoptManager::get_hpopt_total_samples()                     { return get_config_value<int>(xpr_config, HPOPT_PARAM_ID_BAYESOPT_TOTAL_SAMPLES); }
    int HpoptManager::get_hpopt_init_random_samples()               { return get_config_value<int>(xpr_config, HPOPT_PARAM_ID_BAYESOPT_INIT_RAND_SAMPLES); }
    int HpoptManager::get_hpopt_relearn_freq()                      { return get_config_value<int>(xpr_config, HPOPT_PARAM_ID_BAYESOPT_RELEARN_FREQ); }

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
    void _update_statistics_(const vector<double>& evals, double& mean_eval, double& std_eval, double& std_mean_eval)
    {
        if (evals.size() < 2)
        {
            mean_eval = (evals.size() == 1) ? evals[0] : 0.0;
            std_eval = std::numeric_limits<double>::max();
            std_mean_eval = std::numeric_limits<double>::max();
            return;
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
    double HpoptManager::evaluateSample(const bayesopt::vectord& query)
    {
        int repeats_run = 0;
        vector<double> evals;
        double mean_eval = 0.0;
        double std_eval = 0.0;
        double std_mean_eval = 0.0;

        int min_repeats = get_hpopt_min_repeats();
        double estimate_confidence_threshold = get_hpopt_estimate_confidence_threshold();

        // get run manager with params corresponding to this query
        shared_ptr<RunManager> sampled_run_manager = get_run_manager_for_query(query);

        cout << "hp_opt_iter:" << hp_opt_iter << ", query_vector:" << query << endl;
        cout << "sampled_params:" << sampled_run_manager->get_params_string_helper() << endl;

        // run evals
        while (repeats_run < min_repeats || std_mean_eval > estimate_confidence_threshold)
        {
            double eval = thts::run_searches(*sampled_run_manager, true, false);

            evals.push_back(eval);
            _update_statistics_(evals, mean_eval, std_eval, std_mean_eval);
            repeats_run++;

            cout << "Run#=" << repeats_run 
                << ", mean_eval=" << mean_eval 
                << ", std_mean_eval=" << std_mean_eval << " >? " << estimate_confidence_threshold << endl;

            // Early stopping: stop run more than min repeats and cleanly worse than best 
            // i.e. if confidence intervals don't overlap (1.65 std ≈ 95% CI, so <0.1% chance of error)
            constexpr double early_stop_z = 1.65;
            if (repeats_run >= min_repeats && 
                mean_eval + early_stop_z * std_mean_eval < best_mean_eval - early_stop_z * best_std_mean_eval)
            {
                cout << "Early stopping: " << mean_eval << " + " << early_stop_z << "*" << std_mean_eval 
                    << " = " << (mean_eval + early_stop_z * std_mean_eval) 
                    << " < " << (best_mean_eval - early_stop_z * best_std_mean_eval)
                    << " = " << best_mean_eval << " - " << early_stop_z << "*" << best_std_mean_eval << endl;
                break;
            }
        }

        // Update if best eval so far
        if (mean_eval > best_mean_eval)
        {
            this->best_config_map = sampled_run_manager->alg_config;
            this->best_mean_eval = mean_eval;
            this->best_std_mean_eval = std_mean_eval;
        }
        
        // Write to logs
        this->write_hpopt_summary_sample_eval_line(sampled_run_manager, mean_eval, std_mean_eval);

        // Return sample eval
        hp_opt_iter++;
        return -1.0 * mean_eval;
    }

    /**
     * Convert bayesopt sample into a RunManager/ConfigMap to run a search with
     */
    shared_ptr<RunManager> HpoptManager::get_run_manager_for_query(const bayesopt::vectord& query)
    {
        ConfigMap query_xpr_config = get_run_manager_xpr_config_for_query(query);
        ConfigMap query_alg_config = get_run_manager_alg_config_for_query(query);
        return make_shared<RunManager>(xpr_timestamp, query_xpr_config, query_alg_config);
    }


    /**
     * Convert bayesopt sample into a RunManager/ConfigMap to run a search with
     */
    ConfigMap HpoptManager::get_run_manager_xpr_config_for_query(const bayesopt::vectord& query)
    {
        return {
            {XPR_OR_ALG_ID_TAG,                     XPR_PARAMS_ID_TAG},
            {XPR_PARAM_ID_NAME,                     get_config_value<std::string>(xpr_config, XPR_PARAM_ID_NAME)},
            {XPR_PARAM_ID_ENV,                      get_config_value<std::string>(xpr_config, XPR_PARAM_ID_ENV)},
            {XPR_PARAM_ID_MCTS_MODE,                get_config_value<bool>(xpr_config, XPR_PARAM_ID_MCTS_MODE)},
            {XPR_PARAM_ID_GRAPH_SEARCH,             get_config_value<bool>(xpr_config, XPR_PARAM_ID_GRAPH_SEARCH)},
            {XPR_PARAM_ID_MAX_TRIAL_LENGTH,         get_config_value<int>(xpr_config, XPR_PARAM_ID_MAX_TRIAL_LENGTH)},
            {XPR_PARAM_ID_RUNTIME_BOUNDED,          get_config_value<bool>(xpr_config, XPR_PARAM_ID_RUNTIME_BOUNDED)},
            {XPR_PARAM_ID_TERMINATION_BOUND,        get_config_value<double>(xpr_config, XPR_PARAM_ID_TERMINATION_BOUND)},
            {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,    1},
            {XPR_PARAM_ID_SEARCH_THREADS,           get_config_value<int>(xpr_config, XPR_PARAM_ID_SEARCH_THREADS)},
            {XPR_PARAM_ID_EVAL_DELTA,               get_config_value<double>(xpr_config, XPR_PARAM_ID_EVAL_DELTA)},
            {XPR_PARAM_ID_EVAL_ROLLOUTS,            get_config_value<int>(xpr_config, XPR_PARAM_ID_EVAL_ROLLOUTS)},
            {XPR_PARAM_ID_EVAL_THREADS,             get_config_value<int>(xpr_config, XPR_PARAM_ID_EVAL_THREADS)},
        };
    }

    /**
     * Convert bayesopt sample into a RunManager/ConfigMap to run a search with
     */
    ConfigMap HpoptManager::get_run_manager_alg_config_for_query(const bayesopt::vectord& query)
    {
        ConfigMap query_alg_config;
        query_alg_config[XPR_OR_ALG_ID_TAG] = get_config_value<std::string>(alg_config, XPR_OR_ALG_ID_TAG);

        int i = 0;

        for (auto [config_key, value_range] : alg_config)
        {
            if (config_key == XPR_OR_ALG_ID_TAG)
            {
                continue;
            }

            pair<double,double> min_max = std::get<pair<double,double>>(value_range);
            double min = min_max.first;
            double max = min_max.second;

            bool log_scaling = HPOPT_LOG_SCALE_ALG_PARAM_IDS.contains(config_key);
            bool int_param = HPOPT_INT_ALG_PARAM_IDS.contains(config_key);
            
            double query_i = query[i++];
            double sampled_param = get_cts_val_from_bayesopt_sample(query_i, min, max, log_scaling);

            if (!int_param)
            {
                query_alg_config[config_key] = sampled_param;
            }
            else
            {
                query_alg_config[config_key] = get_int_val_from_cts_val(sampled_param, min, max);
            }
        }

        return query_alg_config;
    }

    /**
     * Helper to sample double value using a continuous [0,1] random variable from bayesopt
     * (N.B. we may want to apply log scaling)
     */
    double HpoptManager::get_cts_val_from_bayesopt_sample(double sample_val, double min, double max, bool log_scaling)
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
     * Helper to sample integer value in range [min,max) using a continuous random variable in [min,max]
     */
    int HpoptManager::get_int_val_from_cts_val(double cts_sample, int min, int max)
    {
        if (cts_sample >= max) {
            return max-1;            
        }
        return (int)cts_sample;
    }

    /**
     * Returns if the env we are using is a python env
    */
    bool HpoptManager::is_python_env()
    {
        return (PY_ENVS.contains(get_env_id()) || GYM_ENVS.contains(get_env_id()));
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
        std::filesystem::path dir = filepath.parent_path();

        if (!std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
        }

        // Open the file (will create it if it doesn't exist)
        ofstream file(filepath, ios::out | ios::trunc);
        if (!file.is_open()) 
        {
            throw runtime_error("Failed to open file: " + filepath.string());
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
    void HpoptManager::write_hpopt_summary_header()
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
            // << XPR_PARAM_ID_REPEATED_RUNS_PER_ALG << ","
            << XPR_PARAM_ID_SEARCH_THREADS << ","
            << XPR_PARAM_ID_EVAL_DELTA << ","
            << XPR_PARAM_ID_EVAL_ROLLOUTS << ","
            << XPR_PARAM_ID_EVAL_THREADS << ","
            << HPOPT_PARAM_ID_MIN_REPEATS << ","
            << HPOPT_PARAM_ID_ESTIMATE_CONFIDENCE_THRESHOLD << ","
            << HPOPT_PARAM_ID_BAYESOPT_TOTAL_SAMPLES << ","
            << HPOPT_PARAM_ID_BAYESOPT_INIT_RAND_SAMPLES << ","
            << HPOPT_PARAM_ID_BAYESOPT_RELEARN_FREQ << endl;
        hpopt_summary_fs << get_xpr_name() << ","
            << get_env_id() << ","
            << get_mcts_mode() << ","
            << get_graph_search() << ","
            << get_max_trial_length() << ","
            << xpr_is_runtime_bounded() << ","
            << get_termination_bound() << ","
            // << get_repeated_runs_per_alg() << ","
            << get_num_search_threads() << ","
            << get_eval_delta() << ","
            << get_num_eval_rollouts() << ","
            << get_num_eval_threads() << ","
            << get_hpopt_min_repeats() << "," 
            << get_hpopt_estimate_confidence_threshold() << ","
            << get_hpopt_total_samples() << ","
            << get_hpopt_init_random_samples() << ","
            << get_hpopt_relearn_freq() << endl;

        // Alg level params
        string alg_id = get_alg_id();
        hpopt_summary_fs << endl << alg_id << " params mins/maxs: " << endl << endl;;
        for (const string& alg_param_id : ALG_ID_TO_ALG_PARAM_IDS.at(alg_id))
        {
            pair<double,double> min_max = get_config_value<std::pair<double, double>>(alg_config, alg_param_id);
            hpopt_summary_fs << alg_param_id << " - (min,max,log_scale) = (";
            hpopt_summary_fs << min_max.first << "," << min_max.second << ","
                << HPOPT_LOG_SCALE_ALG_PARAM_IDS.contains(alg_param_id) << ")" << endl;
        }
        
        // csv header for eval lines
        hpopt_summary_fs << endl << "Evaluations:" << endl << endl;
        hpopt_summary_fs << "hp_opt_iter,mean_eval,std_mean_eval,best_eval_so_far";
        for (const string& alg_param_id : ALG_ID_TO_ALG_PARAM_IDS.at(alg_id))
        {
            hpopt_summary_fs << "," << alg_param_id;
        }
        hpopt_summary_fs << endl;
    }

    void HpoptManager::write_hpopt_summary_sample_eval_line(
        shared_ptr<RunManager> run_manager, double mean_eval, double std_mean_eval)
    {
        string alg_id = get_alg_id();
        // Use the alg_config from the run_manager directly
        const ConfigMap& alg_params = run_manager->alg_config;

        hpopt_summary_fs << hp_opt_iter << "," << mean_eval << "," << std_mean_eval << "," << this->best_mean_eval;
        for (const string& alg_param_id : ALG_ID_TO_ALG_PARAM_IDS.at(alg_id))
        {
            // Extract value from variant and print it
            if (alg_params.contains(alg_param_id)) 
            {
                const auto& variant_val = alg_params.at(alg_param_id);
                std::visit([&](auto&& val) {
                    hpopt_summary_fs << "," << val;
                }, variant_val);
            }
        }
        hpopt_summary_fs << endl;
    }
    
    void HpoptManager::write_hpopt_summary_footer()
    {   
        hpopt_summary_fs << endl << "Best Params: " << endl << endl;
        hpopt_summary_fs << "mean_eval - " << best_mean_eval << endl;
        hpopt_summary_fs << "std_mean_eval - " << best_std_mean_eval << endl;
        
        string alg_id = get_alg_id();

        for (const string& alg_param_id : ALG_ID_TO_ALG_PARAM_IDS.at(alg_id))
        {
            // Extract value from variant and print it
            if (best_config_map.contains(alg_param_id)) 
            {
                const auto& variant_val = this->best_config_map.at(alg_param_id);
                std::visit([&](auto&& val) {
                    hpopt_summary_fs << alg_param_id << " - " << val << endl;
                }, variant_val);
            }
        }
        hpopt_summary_fs << endl;
    }
}
