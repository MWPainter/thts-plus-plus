#pragma once

#include "main_mo/run_manager.h"
#include "main_mo/run_xpr.h"

#include "mo/mo_thts_env.h"
#include "mo/mo_thts_manager.h"
#include "mo/mo_thts_decision_node.h"

#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "bayesopt/bayesopt.hpp"
#include "bayesopt/parameters.hpp"

#include "main_mo/configs/hpopt_config.h"


namespace thts {

    /**
     * Struct to cleanly wrap interaction with configs for tuning one algorithm as part of an hyperparamter optimization
     * And the IO/logging associated with running an experiment
     * 
     * Also subclasses bayesopt::ContinuoutModel to inherit hpopt stuff from bayesopt.
     * 
     * Doesnt subclass RunManager, partly out of lazyness to not deal with making the HpoptConfigMap type subclass 
     * ConfigMap. (Basically couldn't quite see how to make the subclassing work).
    */
    struct HpoptManager : public bayesopt::ContinuousModel {
        public:
            std::time_t xpr_timestamp;
            HpoptConfigMap xpr_config;
            HpoptConfigMap alg_config;
            int num_hyperparams;
            ConfigMap best_config_map;
            double best_mean_eval;
            double best_std_mean_eval;
            MoEvalMetrics best_mo_eval_metrics;
            std::ofstream hpopt_summary_fs;
            int hp_opt_iter;
            bayesopt::Parameters bo_params;

            /**
             * Initialised constructor
             * - performs validation on the config to check for user error in the config
            */
            HpoptManager(std::time_t xpr_timestamp, HpoptConfigMap xpr_config, HpoptConfigMap alg_config, bayesopt::Parameters params);

            /**
             * Copy constructor
             */
            HpoptManager(const HpoptManager& other);

            /**
             * Destructor
             */
            ~HpoptManager();

        private:
            /**
             * Checks for enevitable human error in writing the configs
             */
            void validate_config_or_raise_exception();

        public:
            /**
             * Sets the bounding box for bayesopt to sample from
             */
            void set_bayesopt_bounding_box();

            /**
             * Lookup config vector from an xpr_id prefix
             */
            static std::vector<HpoptConfigMap> lookup_config_vector_from_xpr_prefix(std::string xpr_id_prefix);

            /**
             * Config -> BayesOpt params
             */
            static bayesopt::Parameters get_bayesopt_params_from_xpr_config(HpoptConfigMap& xpr_config);

            /**
             * Returns a vector of RunIDs from a vector of ConfigMaps
             * - expects the first ConfigMap to specify the xpr level params
             * - each following ConfigMap specifies and algorithm and corresponding params to run
             */
            static std::shared_ptr<std::vector<HpoptManager>> get_hpopt_managers_from_config_vector(
                std::vector<HpoptConfigMap>& config_vector);

            /**
             * Getters - xpr level config
             */
            std::string get_xpr_name();
            std::string get_env_id();
            bool get_mcts_mode();
            bool get_graph_search();
            bool get_vector_visit_counts();
            int get_max_trial_length();
            bool xpr_is_runtime_bounded();
            double get_termination_bound();
            int get_repeated_runs_per_alg();
            int get_num_search_threads();
            double get_eval_delta();
            int get_num_eval_rollouts();
            int get_num_eval_threads();

            /**
             * Getters - alg level config
             */
            std::string get_alg_id();

            /**
             * Getters - hpopt (xpr) level config
             */
            int get_hpopt_min_repeats();
            double get_hpopt_estimate_confidence_threshold();
            int get_hpopt_total_samples();
            int get_hpopt_init_random_samples();
            int get_hpopt_relearn_freq();

            /**
             * Bayesopt intefrace.
             * Args:
             *  :query: a vector sampled by bayes opt, to return an evaluation score for
             * Returns:
             *  :evaluation_score: a score to be MINIMISED by bayesopt
             */
            virtual double evaluateSample(const bayesopt::vectord& query) override;

            /**
             * Convert bayesopt sample into a RunManager/ConfigMap to run a search with
             */
            std::shared_ptr<RunManager> get_run_manager_for_query(const bayesopt::vectord& query, std::filesystem::path temp_file_path);
            ConfigMap get_run_manager_xpr_config_for_query(const bayesopt::vectord& query);
            ConfigMap get_run_manager_alg_config_for_query(const bayesopt::vectord& query);

            /**
             * Helper to sample continuous value using a continuous [0,1] random variable from bayesopt
             * N.B. our continues values often want to sample with a log scaling
             * Assumes bayesopt sample is from range (log_scaling) ? [log(min),log(max)] : [min,max]
             * Rand returns the sample (log_scaling) ? exp(sample) : sample.
             */
            double get_cts_val_from_bayesopt_sample(double sample_val, double min, double max, bool log_scaling);

            /**
             * Helper to cast a continuous sampled value to an integer
             * Returns an integer in the range [min,max)
             * Sampled by scaling rand to range [min,max] and taking integer portion
             * Cant just cast to int because of the ",max)" edge case
             */
            int get_int_val_from_cts_val(double sample_val, int min, int max);

            /**
             * Returns if the env we are using is a python env
            */
            bool is_python_env();

            /**
             * A unique file for this manager to write hpopt summary to
             */
            std::filesystem::path get_hpopt_summary_filename();
            std::ofstream get_hpopt_summary_filestream();
            void open_hpopt_summary_filestream();
            void close_hpopt_summary_filestream();
            
            /**
             * Functions to write the header, saying the params and ranges being searched over
             * Write eval_lines, for each of the 
             */
            void write_hpopt_summary_header();
            void write_hpopt_summary_sample_eval_line(
                std::shared_ptr<RunManager> run_manager, MoEvalMetrics& mo_eval_metrics);
            void write_hpopt_summary_footer();

            /**
             * Generate a unique temporary file path for a given query vector
             */
            std::filesystem::path get_temp_file_path_for_query(const bayesopt::vectord& query);

            /**
             * Close and delete the current temporary file
             */
            void delete_temp_file(std::filesystem::path temp_file_path);
    };
}