#pragma once

#include "main_aux/run_manager.h"

#include "thts_env.h"
#include "thts_manager.h"
#include "thts_decision_node.h"

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

#include "main_aux/configs/hpopt_config.h"


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
            shared_ptr<ThtsManager> best_thts_manager;
            double best_thts_manager_mean_eval;
            double best_thts_manager_std_mean_eval;
            bool write_eval_logs;
            std::ofstream hpopt_summary_fs;

            /**
             * Initialised constructor
             * - performs validation on the config to check for user error in the config
            */
            HpoptManager(std::time_t xpr_timestamp, ConfigMap xpr_config, ConfigMap alg_config, bayesopt::Parameters params);

        private:
            /**
             * Checks for enevitable human error in writing the configs
             */
            void validate_config_or_raise_exception();

        public:
            /**
             * Lookup config vector from an xpr_id prefix
             */
            static std::vector<HpoptConfigMap> lookup_config_vector_from_xpr_prefix(std::string xpr_id_prefix);

            /**
             * Returns a vector of RunIDs from a vector of ConfigMaps
             * - expects the first ConfigMap to specify the xpr level params
             * - each following ConfigMap specifies and algorithm and corresponding params to run
             */
            static std::shared_ptr<std::vector<HpoptManager>> get_hpopt_managers_from_config_vector(
                std::vector<HpoptConfigMap>& config_vector);

            /**
             * Sets the bounding box for bayesopt to sample from
             */
            void set_bayesopt_bounding_box();

            /**
             * Bayesopt intefrace.
             * Args:
             *  :query: a vector sampled by bayes opt, to return an evaluation score for
             * Returns:
             *  :evaluation_score: a score to be MINIMISED by bayesopt
             */
            virtual double evaluateSample(const bayesopt::vectord& query) override;

        public:
            /**
             * Getters - xpr level config
             */
            std::string get_xpr_name();
            std::string get_env_id();
            bool get_mcts_mode();
            bool get_graph_search();
            int get_max_trial_length();
            bool xpr_is_runtime_bounded();
            double get_termination_bound();
            int get_repeated_runs_per_alg();
            int get_num_search_threads();
            double get_eval_delta();
            int get_num_eval_rollouts();
            int get_num_eval_threads();

            /**
             * Getters - hpopt (xpr) level config
             */
            int get_hpopt_min_repeats();
            double get_hpopt_estimate_confidence_threshold();

            /**
             * Getters - alg level config
             */
            std::string get_alg_id();
            std::pair<double,double> get_bias_range();
            std::pair<int,int> get_uct_budget_range();
            std::pair<double,double> get_init_temp_range();
            std::pair<double,double> get_temp_decay_rate_range();
            std::pair<double,double> get_init_entropy_coeff_range();
            std::pair<double,double> get_entropy_zero_at_range();
            std::pair<double,double> get_epsilon_range();
            std::pair<double,double> get_default_q_value_range();
            
            /**
             * Helper to sample boolean value using a continuous [0,1] random variable from bayesopt
             * Essentially returns (sample_val > 0.5)
             */
            bool get_bool_val_from_cts_sample(double sample_val);

            /**
             * Helper to sample integer value using a continuous [0,1] random variable from bayesopt
             * Returns an integer in the range [min,max)
             * Sampled by scaling rand to range [min,max] and taking integer portion
             */
            int get_int_val_from_cts_sample(double sample_val, int min, int max);

            /**
             * Helper to sample continuous value using a continuous [0,1] random variable from bayesopt
             * N.B. our continues values often want to sample with a log scaling
             * Assumes bayesopt sample is from range (log_scaling) ? [log(min),log(max)] : [min,max]
             * Rand returns the sample (log_scaling) ? exp(sample) : sample.
             */
            double get_cts_val_from_bayesopt_sample(double sample_val, int min, int max, bool log_scaling)

            /**
             * Samplers - alg level config - returns sampled values using [0,1] uniform random sample from bayesopt
             */
            double sample_bias(double rand);
            int sample_uct_budget(double rand);
            double sample_init_temp(double rand);
            double sample_temp_decay_rate(double rand);
            double sample_init_entropy_coeff(double rand);
            double sample_entropy_zero_at(double rand);
            double sample_epsilon(double rand);
            double sample_default_q_value(double rand);

            /**
             * Returns if the env we are using is a python env
            */
            bool is_python_env();

            /**
             * Returns an instance of ThtsEnv to use for this run
            */
            std::shared_ptr<ThtsEnv> get_env();

            /**
             * Creates and returns a thts_manager with params corresponding to bayesopt::vectord query
            */
            std::shared_ptr<ThtsManager> get_thts_manager(std::shared_ptr<ThtsEnv> env, const bayesopt::vectord& query);

            /**
             * Returns a root node to use for search given these params
            */
            std::shared_ptr<ThtsDNode> get_root_search_node(
                std::shared_ptr<ThtsEnv> env, std::shared_ptr<ThtsManager> manager);

            /**
             * Helper to get a config map corresponding to the values
             */
            ConfigMap config_map_from_thts_manager(std::shared_ptr<ThtsManager> manager);

            /**
             * A unique file for this manager to write hpopt summary to
             */
            std::filesystem::path get_hpopt_summary_filename();
            std::ofstream get_hpopt_summary_filestream();
            void open_hpopt_summary_filestream();
            
            /**
             * Functions to write the header, saying the params and ranges being searched over
             * Write eval_lines, for each of the 
             */
            void write_hpopt_summary_header();
            void write_hpopt_summary_sample_eval_line(
                std::shared_ptr<ThtsManager> manager, double mean_eval, double std_mean_eval);
            void write_hpopt_summary_footer();

            /**
             * A unique directory for this manager to write eval logs to
             * And a unique file for logging evals from each run
             */
            std::string get_eval_logs_dir(std::shared_ptr<ThtsManager> manager);
            std::filesystem::path get_eval_log_filename(std::shared_ptr<ThtsManager> manager, int run_idx);
            std::ofstream get_eval_log_filestream(std::shared_ptr<ThtsManager> manager, int run_idx);

            /**
             * Functions for writing to logs files
             */
            void write_eval_log_header(std::ofstream& fs, std::shared_ptr<ThtsManager> manager);
            void write_eval_line(std::ofstream& fs, int run_idx, double eval, double eval_std, int num_trials, double runtime, int num_eval_samples);

        private:
    };
}