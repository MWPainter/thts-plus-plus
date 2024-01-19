#pragma once

#include "thts_chance_node.h"
#include "thts_manager.h"
#include "thts_types.h"
#include "mo/mo_thts_decision_node.h"
#include "mo/mo_thts_manager.h"

#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace thts {
    
    // forward declare
    class MoThtsDNode;
    
    /**
     * An abstract base class for multi objective Chance Nodes.
     */
    class MoThtsCNode : public ThtsCNode {
        // Allow ThtsDNode access to private members
        friend MoThtsDNode;

        public: 
            /**
             * Default constructor.
             * 
             * Initialises the attributes of the class.
             */
            MoThtsCNode(
                std::shared_ptr<MoThtsManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const MoThtsDNode> parent=nullptr);

            /**
             * Mark destructor as virtual for subclassing.
             */
            virtual ~MoThtsCNode() = default;

            /**
             * OVerride final the old backup fn (throws error if try to call)
            */
            virtual void backup_itfc(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                ThtsEnvContext& ctx) override final;

            /**
             * Override of thts backup function for multi objective.
             * 
             * Updates the Sinformation in this node in the backup phase of the thts routine.
             * 
             * Args:
             *      trial_rewards_before_node: 
             *          A list of rewards recieved (at each timestep) on the trial prior to reaching this node.
             *      trial_rewards_after_node:
             *          A list of rewards recieved (at each timestep) on the trial after reaching this node. This list 
             *          includes the reward from R(state,action) that would have been recieved from taking the action 
             *          in this node.
             *      trial_cumulative_return_after_node:
             *          Sum of rewards in the 'trial_rewards_after_node' list
             *      trial_cumulative_return:
             *          Sum of rewards in both of the 'trial_rewards_after_node' and 'trial_rewards_before_node' lists
             */
            virtual void backup_itfc(
                const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
                const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
                const Eigen::ArrayXd trial_cumulative_return_after_node, 
                const Eigen::ArrayXd trial_cumulative_return,
                ThtsEnvContext& ctx) = 0;
    };
}