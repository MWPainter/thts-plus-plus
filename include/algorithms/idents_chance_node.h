#pragma once

#include "algorithms/idents_decision_node.h"
#include "algorithms/idents_manager.h"
#include "thts_types.h"

#include "algorithms/ments_chance_node.h"
#include "algorithms/ments_decision_node.h"
#include "thts_env.h"
#include "thts_env_context.h"
#include "thts_manager.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace thts {
    // forward declare corresponding IDentsDNode class
    class IDentsDNode;
    
    /**
     * An implementation of I(mproved)DENTS in the Thts schema
     * 
     * Just needs to extend the entropy backups
     * 
     * Attributes:
     *      subtree_entropy: The entropy of the policy for the entire subtree
     */
    class IDentsCNode : public MentsCNode {
        // Allow IDentsDNode access to private members
        friend IDentsDNode;

        protected:
            double ments_subtree_entropy;
            double subtree_entropy;

        public: 
            /**
             * Constructor
             */
            IDentsCNode(
                std::shared_ptr<IDentsManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const IDentsDNode> parent=nullptr);

            virtual ~IDentsCNode() = default;

            /**
             * TODO
            */
            void backup_entropy();
            
            /**
             * Implements the thts backup function for the node
             * 
             * Args:
             *      trial_rewards_before_node:  unused
             *      trial_rewards_after_node: unused
             *      trial_cumulative_return_after_node: unused
             *      trial_cumulative_return: unused
             */
            virtual void backup(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                ThtsEnvContext& ctx);

        protected:
            /**
             * Helper to make a IDentsDNode child object.
             */
            std::shared_ptr<IDentsDNode> create_child_node_helper(
                std::shared_ptr<const State> observation, std::shared_ptr<const State> next_state=nullptr) const;
            
        public:
            /**
             * Override create_child_node_helper_itfc still, so that the ThtsCNode create_child_node_itfc function can 
             * create the correct DentsDNode using the above version of create_child_node
             */
            virtual std::shared_ptr<ThtsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation, std::shared_ptr<const State> next_state=nullptr) const;
    };
}