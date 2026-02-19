#pragma once

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

#include "main_aux/configs/xpr_config.h"


namespace thts {

    /**
     * Struct to cleanly wrap interaction with configs for running one algorithm as part of an experiment
     * And the IO/logging associated with running an experiment
    */
    struct RunManager {
        public:
            std::time_t xpr_timestamp;
            ConfigMap xpr_config;
            ConfigMap alg_config;
            std::string xpr_dir_override;  // Optional override for experiment directory name

            /**
             * Initialised constructor
             * - performs validation on the config to check for user error in the config
             * - xpr_dir_override: if non-empty, overrides "get_xpr_name()_timestamp" in get_eval_logs_dir()
            */
            RunManager(
                std::time_t xpr_timestamp, 
                ConfigMap xpr_config, 
                ConfigMap alg_config,
                std::string xpr_dir_override="");

        private:
            /**
             * Checks for enevitable human error in writing the configs
             */
            void validate_config_or_raise_exception();

        public:
            /**
             * Lookup config vector from an xpr_id prefix
             */
            static std::vector<ConfigMap> lookup_config_vector_from_xpr_prefix(std::string xpr_id_prefix);

            /**
             * Returns a vector of RunIDs from a vector of ConfigMaps
             * - expects the first ConfigMap to specify the xpr level params
             * - each following ConfigMap specifies and algorithm and corresponding params to run
             * - xpr_dir_override: if non-empty, overrides the experiment directory name
             */
            static std::shared_ptr<std::vector<RunManager>> get_run_managers_from_config_vector(
                std::vector<ConfigMap>& config_vector,
                std::string xpr_dir_override="");

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
             * Getters - alg level config
             */
            std::string get_alg_id();
            double get_bias();
            int get_uct_budget();
            double get_init_temp();
            double get_temp_decay_rate();
            double get_init_entropy_coeff();
            double get_entropy_zero_at();
            double get_epsilon();
            double get_heuristic_value();

            /**
             * Returns if the env we are using is a python env
            */
            bool is_python_env();

            /**
             * Returns an instance of ThtsEnv to use for this run
            */
            std::shared_ptr<ThtsEnv> get_env();

            /**
             * Returns and instance of ThtsManager to use for this run
            */
            void _add_thts_manager_params_to_args(ThtsManagerArgs& manager_args, std::string env_id);
            std::shared_ptr<ThtsManager> get_thts_manager(std::shared_ptr<ThtsEnv> env);

            /**
             * Returns a root node to use for search given these params
            */
            std::shared_ptr<ThtsDNode> get_root_search_node(
                std::shared_ptr<ThtsEnv> env, std::shared_ptr<ThtsManager> manager);
            
            /**
             * Helper to make a string of of all alg level params
             */
            std::string get_params_string_helper();

            /**
             * A unique results directory for each RunID
             */
            std::string get_eval_logs_dir();
            std::filesystem::path get_eval_log_filename();
            std::ofstream get_eval_log_filestream();

            /**
             * Functions for writing to logs files
             */
            void write_eval_log_header(std::ofstream& fs);
            void write_eval_log_line(
                std::ofstream& fs, 
                int run_idx, 
                double eval, 
                double eval_std, 
                int num_trials, 
                double runtime, 
                double search_budget_consumed, 
                int num_eval_samples);

            /**
             * Filestream to dump tree print outs too
             */
            std::filesystem::path get_tree_log_filename(int run_idx);
            std::ofstream get_tree_log_filestream(int run_idx);
            void dump_tree_log(std::shared_ptr<ThtsDNode> root_node, int run_idx);
    };
}