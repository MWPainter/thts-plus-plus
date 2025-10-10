#pragma once

#include "thts_env.h"
#include "thts_manager.h"
#include "thts_decision_node.h"

#include <ctime>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "bayesopt/bayesopt.hpp"
#include "bayesopt/parameters.hpp"

#include "main_aux/configs/xpr_config.h"


namespace thts {

    /**
     * Struct to cleanly wrap interaction with configs for running one algorithm as part of an experiment
    */
    struct RunManager {
        public:
            std::time_t xpr_timestamp;
            ConfigMap xpr_config;
            ConfigMap alg_config;

            /**
             * Initialised constructor
             * - performs validation on the config to check for user error in the config
            */
            RunID(std::time_t xpr_timestamp, ConfigMap xpr_config, ConfigMap alg_config);

            /**
             * Lookup config vector from an xpr_id prefix
             */
            static std::vector<ConfigMap> lookup_config_vector_from_xpr_prefix(std::string xpr_id_prefix);

            /**
             * Returns a vector of RunIDs from a vector of ConfigMaps
             * - expects the first ConfigMap to specify the xpr level params
             * - each following ConfigMap specifies and algorithm and corresponding params to run
             */
            static std::shared_ptr<std::vector<RunID>> get_run_ids_from_config_vector(
                std::vector<ConfigMap>& config_vector);

            /**
             * Get the algorithm id 
             */
            std::string get_alg_id();

            /**
             * A unique results directory for each RunID
             */
            std::string get_results_dir();

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
            std::shared_ptr<ThtsManager> get_thts_manager(std::shared_ptr<ThtsEnv> env);

            /**
             * Returns a root node to use for search given these params
            */
            std::shared_ptr<ThtsDNode> get_root_search_node(
                std::shared_ptr<ThtsEnv> env, std::shared_ptr<ThtsManager> manager);
    };
}