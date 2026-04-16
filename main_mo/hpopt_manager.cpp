#include "main_mo/hpopt_manager.h"

#include "helper.h"

#include "algorithms/common/decaying_temp.h"

#include "py/pickle_wrapper.h"
#include "py/py_multiprocessing_thts_env.h"
#include "py/gym_multiprocessing_thts_env.h"

#include "main_mo/envs/tree_env.h"
#include "main_mo/envs/test_mo_thts_env.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <set>
#include <type_traits>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <string>
#include <functional>
#include "helper_templates.h"

using namespace std;
using namespace thts;
using namespace thts::python;

namespace py = pybind11;

namespace thts {

    /**
     * Constructor
     */
    HpoptManager::HpoptManager(std::time_t xpr_timestamp, HpoptConfigMap xpr_config, HpoptConfigMap alg_config, bayesopt::Parameters params) :
        bayesopt::ContinuousModel(count_hyperparams(alg_config), params), 
        xpr_timestamp(xpr_timestamp), 
        xpr_config(xpr_config), 
        alg_config(alg_config), 
        num_hyperparams(0),
        hyperparams_optimising(0),
        best_config_map(),
        best_mean_eval(0.0), // hypervolume can never be negative, so set to 0.0
        best_std_mean_eval(0.0),
        best_mo_eval_metrics(),
        hpopt_summary_fs(),
        hp_opt_iter(0),
        bo_params(params)
    {
        validate_config_or_raise_exception();
        for (const auto& [param_id, param_range] : alg_config)
        {
            if (param_id == XPR_OR_ALG_ID_TAG) continue;
            const auto& range = get_config_value<std::pair<double,double>>(alg_config, param_id);
            if (range.first != range.second)
            {
                hyperparams_optimising.push_back(param_id);
                num_hyperparams++;
            }
        }
        if (num_hyperparams == 0)
        {
            throw runtime_error("No hyperparameters to optimise found in alg level config.");
        }
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
        num_hyperparams(other.num_hyperparams),
        hyperparams_optimising(other.hyperparams_optimising),
        best_config_map(other.best_config_map),
        best_mean_eval(other.best_mean_eval),
        best_std_mean_eval(other.best_std_mean_eval),
        best_mo_eval_metrics(other.best_mo_eval_metrics),
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
        if (get_config_value<std::string>(xpr_config, XPR_OR_ALG_ID_TAG) != HPOPT_PARAMS_ID_TAG)
        {
            throw runtime_error("In hpopt manager expecting config entry: {XPR_OR_ALG_ID_TAG,HPOPT_PARAMS_ID_TAG}");
        }

        vector<string> xpr_param_ids = 
        {
            XPR_PARAM_ID_NAME, 
            XPR_PARAM_ID_ENV, 
            // XPR_PARAM_ID_ENV_SIZE,
            XPR_PARAM_ID_MCTS_MODE, 
            XPR_PARAM_ID_GRAPH_SEARCH,
            XPR_PARAM_ID_VECTOR_VISIT_COUNTS,
            XPR_PARAM_ID_MAX_TRIAL_LENGTH,
            XPR_PARAM_ID_HEURISTIC_WEIGHT_GLOBAL,
            XPR_PARAM_ID_HEURISTIC_WEIGHT_LOCAL,
            XPR_PARAM_ID_RUNTIME_BOUNDED, 
            XPR_PARAM_ID_TERMINATION_BOUND, 
            // XPR_PARAM_ID_REPEATED_RUNS_PER_ALG, 
            XPR_PARAM_ID_SEARCH_THREADS, 
            XPR_PARAM_ID_EVAL_DELTA, 
            XPR_PARAM_ID_EVAL_ROLLOUTS, 
            XPR_PARAM_ID_EVAL_THREADS,
            XPR_PARAM_ID_CONVEX_HULL_MAX_SIZE,
            XPR_PARAM_ID_CONVEX_HULL_TOLERANCE,
            XPR_PARAM_ID_USE_SOLVED_LABELLING,
            XPR_PARAM_ID_SOLVED_LABELLING_FAIL_CONFIDENCE,
            XPR_PARAM_ID_SOLVED_LABELLING_TOLERANCE,

            XPR_PARAM_ID_SM_PUSH_RADIUS,
            XPR_PARAM_ID_SM_MAX_NEIGHBOURS_TO_PUSH_TO,
            XPR_PARAM_ID_SM_MIN_SIMPLEX_RADIUS,
            XPR_PARAM_ID_SM_SIMPLEX_SPLIT_COUNTER_THRESHOLD,
            XPR_PARAM_ID_SM_USE_APPROX_NEAREST_VERTEX,
            XPR_PARAM_ID_SM_EVENTUALLY_CONFORMING_SIMPLEX_MAP,
            XPR_PARAM_ID_SM_ALWAYS_ALLOW_NON_CONFORMING_SIMPLEX_TO_SPLIT,

            HPOPT_PARAM_ID_MIN_REPEATS,
            HPOPT_PARAM_ID_ESTIMATE_CONFIDENCE_THRESHOLD,
            HPOPT_PARAM_ID_BAYESOPT_TOTAL_SAMPLES,
            HPOPT_PARAM_ID_BAYESOPT_INIT_RAND_SAMPLES,
            HPOPT_PARAM_ID_BAYESOPT_RELEARN_FREQ,
            HPOPT_PARAM_ID_BAYESOPT_USE_GPML,
            HPOPT_PARAM_ID_USE_HYPERVOLUME_AS_METRIC,
        };

        if (xpr_config.size() != 34)
        {
            throw runtime_error("Expecting 34 entries in the xpr level config.");
        }

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
            ALG_ID_CZT, 
            ALG_ID_CZT_DOUBLING, 
            ALG_ID_CH_UCT, 
            ALG_ID_CH_CZT, 
            ALG_ID_CH_CZT_DOUBLING, 
            ALG_ID_CH_BTS, 
            ALG_ID_CH_DENTS, 
            ALG_ID_CH_HVUCT, 
            ALG_ID_CH_PARETO, 
            ALG_ID_CH_CHEBY,
            ALG_ID_CH_STANDARD_CHEBY,
            ALG_ID_SM_BTS,
            ALG_ID_SM_DENTS,
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
     * Helper to count the number of hyperparams being optimised (where min != max)
     */
    int HpoptManager::count_hyperparams(const HpoptConfigMap& alg_config)
    {
        int count = 0;
        for (const auto& [param_id, param_range] : alg_config)
        {
            if (param_id == XPR_OR_ALG_ID_TAG) continue;
            const auto& range = get_config_value<std::pair<double,double>>(alg_config, param_id);
            if (range.first != range.second) count++;
        }
        return count;
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
        bool bayesopt_use_gpml = get_config_value<int>(xpr_config, HPOPT_PARAM_ID_BAYESOPT_USE_GPML) > 0;

        bayesopt::Parameters bo_params;
        bo_params.surr_name = bayesopt_use_gpml ? "sGaussianProcessML" : "sGaussianProcessNormal";
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
    bool HpoptManager::get_vector_visit_counts()  { return get_config_value<bool>(xpr_config, XPR_PARAM_ID_VECTOR_VISIT_COUNTS); }
    int HpoptManager::get_max_trial_length()      { return get_config_value<int>(xpr_config, XPR_PARAM_ID_MAX_TRIAL_LENGTH); }
    double HpoptManager::get_heuristic_weight_global() { return get_config_value<double>(xpr_config, XPR_PARAM_ID_HEURISTIC_WEIGHT_GLOBAL); }
    double HpoptManager::get_heuristic_weight_local() { return get_config_value<double>(xpr_config, XPR_PARAM_ID_HEURISTIC_WEIGHT_LOCAL); }
    bool HpoptManager::xpr_is_runtime_bounded()   { return get_config_value<bool>(xpr_config, XPR_PARAM_ID_RUNTIME_BOUNDED); }
    double HpoptManager::get_termination_bound()  { return get_config_value<double>(xpr_config, XPR_PARAM_ID_TERMINATION_BOUND); }
    int HpoptManager::get_repeated_runs_per_alg() { return get_config_value<int>(xpr_config, XPR_PARAM_ID_REPEATED_RUNS_PER_ALG); }
    int HpoptManager::get_num_search_threads()    { return get_config_value<int>(xpr_config, XPR_PARAM_ID_SEARCH_THREADS); }
    double HpoptManager::get_eval_delta()         { return get_config_value<double>(xpr_config, XPR_PARAM_ID_EVAL_DELTA); }
    int HpoptManager::get_num_eval_rollouts()     { return get_config_value<int>(xpr_config, XPR_PARAM_ID_EVAL_ROLLOUTS); }
    int HpoptManager::get_num_eval_threads()      { return get_config_value<int>(xpr_config, XPR_PARAM_ID_EVAL_THREADS); }
    int HpoptManager::get_convex_hull_max_size()  { return get_config_value<int>(xpr_config, XPR_PARAM_ID_CONVEX_HULL_MAX_SIZE); }
    double HpoptManager::get_convex_hull_tolerance() { return get_config_value<double>(xpr_config, XPR_PARAM_ID_CONVEX_HULL_TOLERANCE); }
    bool HpoptManager::get_use_solved_labelling() { return get_config_value<bool>(xpr_config, XPR_PARAM_ID_USE_SOLVED_LABELLING); }
    double HpoptManager::get_solved_labelling_fail_confidence() { return get_config_value<double>(xpr_config, XPR_PARAM_ID_SOLVED_LABELLING_FAIL_CONFIDENCE); }
    double HpoptManager::get_solved_labelling_tolerance() { return get_config_value<double>(xpr_config, XPR_PARAM_ID_SOLVED_LABELLING_TOLERANCE); }

    /**
     * Getters - sm level config
     */
    int HpoptManager::get_sm_push_radius()                 { return get_config_value<int>(xpr_config, XPR_PARAM_ID_SM_PUSH_RADIUS); }
    int HpoptManager::get_sm_max_neighbours_to_push_to()   { return get_config_value<int>(xpr_config, XPR_PARAM_ID_SM_MAX_NEIGHBOURS_TO_PUSH_TO); }
    double HpoptManager::get_sm_min_simplex_radius()        { return get_config_value<double>(xpr_config, XPR_PARAM_ID_SM_MIN_SIMPLEX_RADIUS); }
    int HpoptManager::get_sm_simplex_split_counter_threshold() { return get_config_value<int>(xpr_config, XPR_PARAM_ID_SM_SIMPLEX_SPLIT_COUNTER_THRESHOLD); }
    bool HpoptManager::get_use_approx_nearest_vertex()     { return get_config_value<bool>(xpr_config, XPR_PARAM_ID_SM_USE_APPROX_NEAREST_VERTEX); }
    bool HpoptManager::get_eventually_conforming_simplex_map() { return get_config_value<bool>(xpr_config, XPR_PARAM_ID_SM_EVENTUALLY_CONFORMING_SIMPLEX_MAP); }
    bool HpoptManager::get_always_allow_non_conforming_simplex_to_split() { return get_config_value<bool>(xpr_config, XPR_PARAM_ID_SM_ALWAYS_ALLOW_NON_CONFORMING_SIMPLEX_TO_SPLIT); }

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
    bool HpoptManager::get_hpopt_use_hypervolume_as_metric() { return get_config_value<bool>(xpr_config, HPOPT_PARAM_ID_USE_HYPERVOLUME_AS_METRIC); }

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
        bool use_hypervolume_as_metric = get_hpopt_use_hypervolume_as_metric();
        int repeats_run = 0;
        vector<double> evals;
        double mean_eval = 0.0;
        double std_eval = 0.0;
        double std_mean_eval = 0.0;
        MoEvalMetrics avg_mo_eval_metrics = MoEvalMetrics();

        int min_repeats = get_hpopt_min_repeats();
        double estimate_confidence_threshold = get_hpopt_estimate_confidence_threshold();

        // create temp file for this query
        filesystem::path temp_file_path = get_temp_file_path_for_query(query);

        // get run manager with params corresponding to this query
        shared_ptr<RunManager> sampled_run_manager = get_run_manager_for_query(query, temp_file_path);

        cout << "hp_opt_iter:" << hp_opt_iter << ", query_vector:" << query << endl;
        cout << "sampled_params:" << sampled_run_manager->get_params_string_helper() << endl;

        // run evals
        while (repeats_run < min_repeats || std_mean_eval > estimate_confidence_threshold)
        {
            MoEvalMetrics mo_eval_metrics = thts::run_searches(*sampled_run_manager, true, false, false);

            avg_mo_eval_metrics.ctx_mean *= repeats_run / (double)(repeats_run + 1);
            avg_mo_eval_metrics.reweighted_ctx_mean *= repeats_run / (double)(repeats_run + 1);
            avg_mo_eval_metrics.normalised_ctx_mean *= repeats_run / (double)(repeats_run + 1);
            avg_mo_eval_metrics.hypervolume *= repeats_run / (double)(repeats_run + 1);
            avg_mo_eval_metrics.additive_eps_metric *= repeats_run / (double)(repeats_run + 1);
            avg_mo_eval_metrics.sparsity_metric *= repeats_run / (double)(repeats_run + 1);
            avg_mo_eval_metrics.normalised_hypervolume *= repeats_run / (double)(repeats_run + 1);
            avg_mo_eval_metrics.normalised_additive_eps_metric *= repeats_run / (double)(repeats_run + 1);
            avg_mo_eval_metrics.normalised_sparsity_metric *= repeats_run / (double)(repeats_run + 1);

            avg_mo_eval_metrics.ctx_mean += mo_eval_metrics.ctx_mean / (double)(repeats_run + 1);
            avg_mo_eval_metrics.reweighted_ctx_mean += mo_eval_metrics.reweighted_ctx_mean / (double)(repeats_run + 1);
            avg_mo_eval_metrics.normalised_ctx_mean += mo_eval_metrics.normalised_ctx_mean / (double)(repeats_run + 1);
            avg_mo_eval_metrics.hypervolume += mo_eval_metrics.hypervolume / (double)(repeats_run + 1);
            avg_mo_eval_metrics.additive_eps_metric += mo_eval_metrics.additive_eps_metric / (double)(repeats_run + 1);
            avg_mo_eval_metrics.sparsity_metric += mo_eval_metrics.sparsity_metric / (double)(repeats_run + 1);
            avg_mo_eval_metrics.normalised_hypervolume += mo_eval_metrics.normalised_hypervolume / (double)(repeats_run + 1);
            avg_mo_eval_metrics.normalised_additive_eps_metric += mo_eval_metrics.normalised_additive_eps_metric / (double)(repeats_run + 1);
            avg_mo_eval_metrics.normalised_sparsity_metric += mo_eval_metrics.normalised_sparsity_metric / (double)(repeats_run + 1);
            
            // Chose normalised hypervolume as eval metric
            // This should have much lower variance than ctx_mean, as it doesn't depend on sampled outcomes that may 
            // leave the tree and resort to following random policies

            double eval = mo_eval_metrics.normalised_hypervolume;
            if (!use_hypervolume_as_metric) 
            {
                eval = mo_eval_metrics.normalised_ctx_mean;
            }
            evals.push_back(eval);
            _update_statistics_(evals, mean_eval, std_eval, std_mean_eval);
            repeats_run++;

            cout << "Run#=" << repeats_run 
                << ", mean_eval=" << mean_eval 
                << ", std_mean_eval=" << std_mean_eval << " >? " << estimate_confidence_threshold << endl;

            // Early stopping: stop run more than min repeats and clearly worse than best 
            // i.e. if confidence intervals don't overlap (1.65 std ≈ 95% CI, so <0.1% chance of error)
            constexpr double early_stop_z = 1.65;
            if (repeats_run >= min_repeats && 
                best_mean_eval != std::numeric_limits<double>::lowest() &&
                mean_eval + early_stop_z * std_mean_eval < best_mean_eval - early_stop_z * best_std_mean_eval)
            {
                cout << "Early stopping: " << mean_eval << " + " << early_stop_z << "*" << std_mean_eval 
                    << " = " << (mean_eval + early_stop_z * std_mean_eval) 
                    << " < " << (best_mean_eval - early_stop_z * best_std_mean_eval)
                    << " = " << best_mean_eval << " - " << early_stop_z << "*" << best_std_mean_eval << endl;
                break;
            }
        }

        // close temp file for this query
        delete_temp_file(temp_file_path);

        // Update if best eval so far
        if (mean_eval > best_mean_eval)
        {
            this->best_config_map = sampled_run_manager->alg_config;
            this->best_mean_eval = mean_eval;
            this->best_std_mean_eval = std_mean_eval;
            this->best_mo_eval_metrics = avg_mo_eval_metrics;
        }
        
        // Write to logs
        this->write_hpopt_summary_sample_eval_line(sampled_run_manager, avg_mo_eval_metrics);

        // Return sample eval
        hp_opt_iter++;
        return -1.0 * mean_eval;
    }

    /**
     * Convert bayesopt sample into a RunManager/ConfigMap to run a search with
     */
    shared_ptr<RunManager> HpoptManager::get_run_manager_for_query(const bayesopt::vectord& query, filesystem::path temp_file_path)
    {
        ConfigMap query_xpr_config = get_run_manager_xpr_config_for_query(query);
        ConfigMap query_alg_config = get_run_manager_alg_config_for_query(query);
        string thts_unique_filename = temp_file_path.string();
        return make_shared<RunManager>(xpr_timestamp, query_xpr_config, query_alg_config, "", thts_unique_filename);
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
            {XPR_PARAM_ID_ENV_SIZE,                 NO_ENV_SIZE},
            {XPR_PARAM_ID_MCTS_MODE,                get_config_value<bool>(xpr_config, XPR_PARAM_ID_MCTS_MODE)},
            {XPR_PARAM_ID_GRAPH_SEARCH,             get_config_value<bool>(xpr_config, XPR_PARAM_ID_GRAPH_SEARCH)},
            {XPR_PARAM_ID_VECTOR_VISIT_COUNTS,      get_config_value<bool>(xpr_config, XPR_PARAM_ID_VECTOR_VISIT_COUNTS)},
            {XPR_PARAM_ID_MAX_TRIAL_LENGTH,         get_config_value<int>(xpr_config, XPR_PARAM_ID_MAX_TRIAL_LENGTH)},
            {XPR_PARAM_ID_HEURISTIC_WEIGHT_GLOBAL,  get_config_value<double>(xpr_config, XPR_PARAM_ID_HEURISTIC_WEIGHT_GLOBAL)},
            {XPR_PARAM_ID_HEURISTIC_WEIGHT_LOCAL,   get_config_value<double>(xpr_config, XPR_PARAM_ID_HEURISTIC_WEIGHT_LOCAL)},
            {XPR_PARAM_ID_RUNTIME_BOUNDED,          get_config_value<bool>(xpr_config, XPR_PARAM_ID_RUNTIME_BOUNDED)},
            {XPR_PARAM_ID_TERMINATION_BOUND,        get_config_value<double>(xpr_config, XPR_PARAM_ID_TERMINATION_BOUND)},
            {XPR_PARAM_ID_REPEATED_RUNS_PER_ALG,    1},
            {XPR_PARAM_ID_SEARCH_THREADS,           get_config_value<int>(xpr_config, XPR_PARAM_ID_SEARCH_THREADS)},
            {XPR_PARAM_ID_EVAL_DELTA,               get_config_value<double>(xpr_config, XPR_PARAM_ID_EVAL_DELTA)},
            {XPR_PARAM_ID_EVAL_ROLLOUTS,            get_config_value<int>(xpr_config, XPR_PARAM_ID_EVAL_ROLLOUTS)},
            {XPR_PARAM_ID_EVAL_THREADS,             get_config_value<int>(xpr_config, XPR_PARAM_ID_EVAL_THREADS)},
            {XPR_PARAM_ID_CONVEX_HULL_MAX_SIZE,     get_config_value<int>(xpr_config, XPR_PARAM_ID_CONVEX_HULL_MAX_SIZE)},
            {XPR_PARAM_ID_CONVEX_HULL_TOLERANCE,    get_config_value<double>(xpr_config, XPR_PARAM_ID_CONVEX_HULL_TOLERANCE)},

            {XPR_PARAM_ID_USE_SOLVED_LABELLING,             get_config_value<bool>(xpr_config, XPR_PARAM_ID_USE_SOLVED_LABELLING)},
            {XPR_PARAM_ID_SOLVED_LABELLING_FAIL_CONFIDENCE, get_config_value<double>(xpr_config, XPR_PARAM_ID_SOLVED_LABELLING_FAIL_CONFIDENCE)},
            {XPR_PARAM_ID_SOLVED_LABELLING_TOLERANCE,       get_config_value<double>(xpr_config, XPR_PARAM_ID_SOLVED_LABELLING_TOLERANCE)},

            {XPR_PARAM_ID_SM_PUSH_RADIUS,                                   get_config_value<int>(xpr_config, XPR_PARAM_ID_SM_PUSH_RADIUS)},
            {XPR_PARAM_ID_SM_MAX_NEIGHBOURS_TO_PUSH_TO,                     get_config_value<int>(xpr_config, XPR_PARAM_ID_SM_MAX_NEIGHBOURS_TO_PUSH_TO)},
            {XPR_PARAM_ID_SM_MIN_SIMPLEX_RADIUS,                            get_config_value<double>(xpr_config, XPR_PARAM_ID_SM_MIN_SIMPLEX_RADIUS)},
            {XPR_PARAM_ID_SM_SIMPLEX_SPLIT_COUNTER_THRESHOLD,               get_config_value<int>(xpr_config, XPR_PARAM_ID_SM_SIMPLEX_SPLIT_COUNTER_THRESHOLD)},
            {XPR_PARAM_ID_SM_USE_APPROX_NEAREST_VERTEX,                     get_config_value<bool>(xpr_config, XPR_PARAM_ID_SM_USE_APPROX_NEAREST_VERTEX)},
            {XPR_PARAM_ID_SM_EVENTUALLY_CONFORMING_SIMPLEX_MAP,             get_config_value<bool>(xpr_config, XPR_PARAM_ID_SM_EVENTUALLY_CONFORMING_SIMPLEX_MAP)},
            {XPR_PARAM_ID_SM_ALWAYS_ALLOW_NON_CONFORMING_SIMPLEX_TO_SPLIT,  get_config_value<bool>(xpr_config, XPR_PARAM_ID_SM_ALWAYS_ALLOW_NON_CONFORMING_SIMPLEX_TO_SPLIT)},
        };
    }

    /**
     * Convert bayesopt sample into a RunManager/ConfigMap to run a search with
     */
    ConfigMap HpoptManager::get_run_manager_alg_config_for_query(const bayesopt::vectord& query)
    {
        ConfigMap query_alg_config;
        query_alg_config[XPR_OR_ALG_ID_TAG] = get_config_value<std::string>(alg_config, XPR_OR_ALG_ID_TAG);

        // Set all fixed params (where min == max) to their fixed value
        for (const auto& [config_key, value_range] : alg_config)
        {
            if (config_key == XPR_OR_ALG_ID_TAG) continue;
            pair<double,double> min_max = get_config_value<std::pair<double,double>>(alg_config, config_key);
            if (min_max.first == min_max.second)
            {
                query_alg_config[config_key] = min_max.first;
            }
        }

        // Sample optimised hyperparams from the query vector
        for (int i = 0; i < num_hyperparams; i++)
        {
            const string& config_key = hyperparams_optimising[i];
            pair<double,double> min_max = get_config_value<std::pair<double,double>>(alg_config, config_key);
            double min = min_max.first;
            double max = min_max.second;

            bool log_scaling = HPOPT_LOG_SCALE_ALG_PARAM_IDS.contains(config_key);
            bool int_param = HPOPT_INT_ALG_PARAM_IDS.contains(config_key);

            double sampled_param = get_cts_val_from_bayesopt_sample(query[i], min, max, log_scaling);

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
        ss << "mo_hpopt_summaries/" << get_xpr_name() << "_" << xpr_timestamp << "_alg_" << get_alg_id() << ".txt";
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
        ofstream file(filepath, ios::out | ios::app);
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
        // If appending to an existing file, avoid duplicating the header.
        if (hpopt_summary_fs.tellp() != std::streampos(0)) {
            return;
        }

        // Xpr level params
        hpopt_summary_fs << "Hpopt Xpr level params:" << endl << endl;;
        hpopt_summary_fs << XPR_PARAM_ID_NAME << ","
            << XPR_PARAM_ID_ENV << ","
            << XPR_PARAM_ID_MCTS_MODE << ","
            << XPR_PARAM_ID_GRAPH_SEARCH << ","
            << XPR_PARAM_ID_VECTOR_VISIT_COUNTS << ","
            << XPR_PARAM_ID_MAX_TRIAL_LENGTH << ","
            << XPR_PARAM_ID_HEURISTIC_WEIGHT_GLOBAL << ","
            << XPR_PARAM_ID_HEURISTIC_WEIGHT_LOCAL << ","
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
            << get_vector_visit_counts() << ","
            << get_max_trial_length() << ","
            << get_heuristic_weight_global() << ","
            << get_heuristic_weight_local() << ","
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
        hpopt_summary_fs << "hp_opt_iter,"
            << "mean_eval,"
            << "std_mean_eval,"
            << "best_eval_so_far,"
            << "ctx_mean,"
            << "reweighted_ctx_mean,"
            << "normalised_ctx_mean,"
            << "hypervolume,"
            << "normalised_hypervolume";
        for (const string& alg_param_id : ALG_ID_TO_ALG_PARAM_IDS.at(alg_id))
        {
            hpopt_summary_fs << "," << alg_param_id;
        }
        hpopt_summary_fs << endl;
    }

    void HpoptManager::write_hpopt_summary_sample_eval_line(
        shared_ptr<RunManager> run_manager, MoEvalMetrics& mo_eval_metrics)
    {
        string alg_id = get_alg_id();
        // Use the alg_config from the run_manager directly
        const ConfigMap& alg_params = run_manager->alg_config;

        hpopt_summary_fs 
            << hp_opt_iter << "," 
            << mo_eval_metrics.ctx_mean << "," 
            << mo_eval_metrics.reweighted_ctx_mean << "," 
            << mo_eval_metrics.normalised_ctx_mean << "," 
            << mo_eval_metrics.hypervolume << "," 
            << mo_eval_metrics.normalised_hypervolume;
        for (const string& alg_param_id : ALG_ID_TO_ALG_PARAM_IDS.at(alg_id))
        {
            // Extract value from variant and print it
            if (alg_params.contains(alg_param_id)) 
            {
                const auto& variant_val = alg_params.at(alg_param_id);
                std::visit([&](auto&& val) {
                    using T = std::decay_t<decltype(val)>;
                    if constexpr (std::is_same_v<T, std::vector<int>>) {
                        throw runtime_error("Trying to print vector<int> to hpopt summary, and should never happen");
                    } else {
                        hpopt_summary_fs << "," << val;
                    }
                }, variant_val);
            }
        }
        hpopt_summary_fs << endl;
    }
    
    void HpoptManager::write_hpopt_summary_footer()
    {   
        hpopt_summary_fs << endl;
        hpopt_summary_fs << "Best Params Eval: " << endl << endl;
        hpopt_summary_fs << "mean_eval - " << best_mean_eval << endl;
        hpopt_summary_fs << "std_mean_eval - " << best_std_mean_eval << endl;
        hpopt_summary_fs << endl;

        hpopt_summary_fs << "Best Eval Metrics: " << endl << endl;
        hpopt_summary_fs << "ctx_mean - " << best_mo_eval_metrics.ctx_mean << endl;
        hpopt_summary_fs << "reweighted_ctx_mean - " << best_mo_eval_metrics.reweighted_ctx_mean << endl;
        hpopt_summary_fs << "normalised_ctx_mean - " << best_mo_eval_metrics.normalised_ctx_mean << endl;
        hpopt_summary_fs << "hypervolume - " << best_mo_eval_metrics.hypervolume << endl;
        hpopt_summary_fs << "normalised_hypervolume - " << best_mo_eval_metrics.normalised_hypervolume << endl;
        hpopt_summary_fs << endl;

        hpopt_summary_fs << "Best Params: " << endl << endl;
        string alg_id = get_alg_id();

        for (const string& alg_param_id : ALG_ID_TO_ALG_PARAM_IDS.at(alg_id))
        {
            // Extract value from variant and print it
            if (best_config_map.contains(alg_param_id)) 
            {
                const auto& variant_val = this->best_config_map.at(alg_param_id);
                std::visit([&](auto&& val) {
                    using T = std::decay_t<decltype(val)>;
                    if constexpr (std::is_same_v<T, std::vector<int>>) {
                        throw runtime_error("Trying to print vector<int> to hpopt summary, and should never happen");
                    } else {
                        hpopt_summary_fs << alg_param_id << " - " << val << endl;
                    }
                }, variant_val);
            }
        }
        hpopt_summary_fs << endl;
    }

    /**
     * Generate a unique temporary file path for a given query vector
     * Uses a hash of the query vector values to create a unique identifier
     */
    std::filesystem::path HpoptManager::get_temp_file_path_for_query(const bayesopt::vectord& query)
    {
        // Create a hash from the query vector values
        size_t query_hash = 0;
        for (size_t i = 0; i < query.size(); ++i) {
            query_hash = thts::helper::hash_combine(query_hash, query[i]);
        }
        
        // Also incorporate xpr_timestamp and alg_id for additional uniqueness
        query_hash = thts::helper::hash_combine(query_hash, xpr_timestamp);
        std::hash<std::string> string_hasher;
        query_hash = thts::helper::hash_combine(query_hash, string_hasher(get_alg_id()));
        
        // Create unique filename using hash and ensure it's in a temp directory
        stringstream ss;
        ss << "tmp/" << get_xpr_name() << "_" << xpr_timestamp << "_alg_" << get_alg_id() 
           << "_query_" << std::hex << query_hash << ".tmp";
        
        std::filesystem::path filepath(ss.str());
        
        // Create parent directory if it doesn't exist
        std::filesystem::path dir = filepath.parent_path();
        if (!std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
        }
        
        // Create the file and write "Urgh!" to it, then close immediately
        std::ofstream file(filepath, std::ios::out | std::ios::trunc);
        if (file.is_open()) {
            file << "Urgh!" << std::endl;
            file.close();
        } else {
            throw runtime_error("Failed to create temp file: " + filepath.string());
        }
        
        return filepath;
    }

    /**
     * Close and delete the current temporary file
     */
    void HpoptManager::delete_temp_file(filesystem::path temp_file_path)
    {
        if (!temp_file_path.empty() && std::filesystem::exists(temp_file_path)) {
            std::filesystem::remove(temp_file_path);
        }
    }
}
