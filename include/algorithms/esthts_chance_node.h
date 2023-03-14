#pragma once

#include "algorithms/esthts_decision_node.h"
#include "algorithms/ments_manager.h"
#include "thts_types.h"

#include "algorithms/dbments_chance_node.h"
#include "algorithms/dbments_decision_node.h"
#include "thts_env.h"
#include "thts_env_context.h"
#include "thts_manager.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace thts {
    // forward declare corresponding ESThtsDNode class
    class ESThtsDNode;
    
    /**
     * Implementation of Boltzmann search Thts
     * 
     */
    class ESThtsCNode : public DBMentsCNode {
        // Allow ESThtsDNode access to private members
        friend ESThtsDNode;

        protected:

        public: 
            /**
             * Constructor
             */
            ESThtsCNode(
                std::shared_ptr<MentsManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const ESThtsDNode> parent=nullptr);

            virtual ~ESThtsCNode() = default;

            /**
             * Calls both the entropy backup and dp backup from DPCNode
             */
            virtual void backup(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                ThtsEnvContext& ctx);

        protected:
            /**
             * Helper to make a ESThtsDNode child object.
             */
            std::shared_ptr<ESThtsDNode> create_child_node_helper(
                std::shared_ptr<const State> observation, std::shared_ptr<const State> next_state=nullptr) const;

            /**
             * Override the pretty print to print out both the dp value and the soft value
             */
            virtual std::string get_pretty_print_val() const;
            
        public:
            /**
             * Override create_child_node_helper_itfc still, so that the ThtsCNode create_child_node_itfc function can 
             * create the correct ESThtsDNode using the above version of create_child_node
             */
            virtual std::shared_ptr<ThtsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation, std::shared_ptr<const State> next_state=nullptr) const;
    };
}