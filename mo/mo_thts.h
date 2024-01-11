#pragma once

#include "thts.h"
#include "thts_env_context.h"
#include "thts_logger.h"
#include "thts_manager.h"
#include "mo/mo_thts_decision_node.h"
#include "mo/mo_thts_env.h"

#include <chrono>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <Eigen/Dense>


namespace thts {
    /**
     * A class encapsulating all of the logic required to run a thts routine on a multi-objective environment.
     * 
     * See ThtsPool for a full description of this class
     * 
     * This multi-objective version just replaces the 'double' reward type with rewards of type 'Eigen::ArrayXd' and 
     * makes appropriate type changes to MultiObjective (Mo____) classes where necessary.
     */
    class MoThtsPool : public ThtsPool {

        public:
            /**
             * Constructs the MoThtsPool with 'num_threads' worker threads.
             * 
             * Args:
             *      manager: The ThtsManager to use for this instance of thts
             *      root_node: 
             *          The root node to run thts on. The algorithm is specified by the subclass of ThtsDNode used. If 
             *          NULL, then a default root node construction is attempted using the initial state from 
             *          thts_manager->thts_env.
             *      num_threads: The number of worker threads to spawn
             */
            MoThtsPool(
                std::shared_ptr<ThtsManager> thts_manager=nullptr, 
                std::shared_ptr<MoThtsDNode> root_node=nullptr, 
                int num_threads=1,
                std::shared_ptr<ThtsLogger> logger=nullptr);

            /**
             * Destructor. (Should just call ~ThtsPool()).
             */
            virtual ~MoThtsPool() = default;

        protected:

            /**
             * Runs the selection phase of a trial, called by worker threads.
             * 
             * Duplicate of ThtsPool::run_backup_phase, but updating types to use VectorXd rewards, rather than double.
             * This also needs to call get_mo_reward_itfc, rather than get_reward_itfc
             * 
             * Ideally we would have a common implementation (using templates somehow), but it seems to make the code a 
             * bit obfuscated. So just accepting duplicated code for now :/
             */
            void run_selection_phase(
                std::vector<std::pair<std::shared_ptr<ThtsDNode>,std::shared_ptr<ThtsCNode>>>& nodes_to_backup, 
                std::vector<Eigen::ArrayXd>& rewards, 
                ThtsEnvContext& context,
                int tid);

            /**
             * Runs the backup phase of a trial, called by worker threads.
             * 
             * Duplicate of ThtsPool::run_backup_phase, but updating types to use VectorXd rewards, rather than double
             * 
             * Ideally we would have a common implementation (using templates somehow), but it seems to make the code a 
             * bit obfuscated. So just accepting duplicated code for now :/
             */
            void run_backup_phase(
                std::vector<std::pair<std::shared_ptr<ThtsDNode>,std::shared_ptr<ThtsCNode>>>& nodes_to_backup, 
                std::vector<Eigen::ArrayXd>& rewards, 
                ThtsEnvContext& context);

            /**
             * Performs a single thts trial. Called by worker_fn.
             * 
             * Duplicates ThtsPool::run_thts_trial, but updates types to use VectorXd rewards.
             * 
             * Ideally we would have a common implementation (using templates somehow), but it seems to make the code a 
             * bit obfuscated. So just accepting duplicated code for now :/
             */
            virtual void run_thts_trial(int trials_remaining, int tid) override;          
    };
}