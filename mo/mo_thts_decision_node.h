#pragma once

#include "thts_decision_node.h"
#include "thts_manager.h"
#include "mo/mo_thts_chance_node.h"
#include "mo/mo_thts_manager.h"

#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

namespace thts {
    
    // forward declare
    class MoThtsCNode;
    class ThtsLogger;
    class MoThtsPool;
    class MoThtsContext;

    /**
     * An abstract base class for a multi objective Decision Node.
     * 
     * Note that this class has a scalar 'heuristic_value' and a 'mo_heuristic_value'. Decided it's not worth 
     * seperating it out into DNode (without 'heuristic_value') and ThtsDNode for scalar objectives for now. If 
     * multi-objective methods become more mainstream then it might be worth optimising the library for memory for 
     * multi-objective algorithms, but this will be very marginal anyway.
     * 
     * Member variables:
     *      mo_heuristic_value:
     *          The (multi objective) heuristic value of this decision node
     */
    class MoThtsDNode : public ThtsDNode {
        // Allow ThtsCNode, Logger and Pool access to private members
        friend MoThtsCNode;
        friend ThtsLogger;
        friend MoThtsPool;
        friend MoThtsContext;

        protected:
            Eigen::ArrayXd mo_heuristic_value;

        public: 
            /**
             * Constructor.
             * 
             * Initialises the attributes of the class.
             */
            MoThtsDNode(
                std::shared_ptr<MoThtsManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const MoThtsCNode> parent=nullptr); 

            /**
             * Mark destructor as virtual for subclassing.
             */
            virtual ~MoThtsDNode() = default;

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
             * Updates the information in this node in the backup phase of the thts routine.
             * 
             * Args:
             *      trial_rewards_before_node: 
             *          A list of rewards recieved (at each timestep) on the trial prior to reaching this node.
             *      trial_rewards_after_node:
             *          A list of rewards recieved (at each timestep) on the trial after reaching this node. This list 
             *          includes the reward from R(state,action) that would have been recieved from taking an action 
             *          from this node.
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