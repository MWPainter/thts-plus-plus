#pragma once

#include "algorithms/esthts_chance_node.h"
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
    class ESThtsCNode;

    /**
     * Implementation of Boltzmann search Thts, built ontop of DBMents because that's already coded up
     * 
     */
    class ESThtsDNode : public DBMentsDNode {
        // Allow IDBMentsDNode access to private members
        friend ESThtsCNode;

        public:  
            /**
             * Constructor
             */
            ESThtsDNode(
                std::shared_ptr<MentsManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const ESThtsCNode> parent=nullptr); 

            virtual ~ESThtsDNode() = default;          

            /**
             * To implement ESThts given an implementation of ments, we just need to swap out the soft q value to 
             * return the dp backup value
             * 
             * Args:
             *      action: 
             *          The action to get the corresponding q value for
             *      opponent_coeff: 
             *          A value of -1.0 or 1.0 for if we are acting as the opponent in a two player game or not 
             *          respectively
             */
            virtual double get_soft_q_value(std::shared_ptr<const Action> action, double opponent_coeff) const;
            
            /**
             * Calls both the entropy backup and dp backup from DPDNode
             */
            virtual void backup(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                ThtsEnvContext& ctx);

        protected:
            /**
             * Helper to make a ESThtsCNode child object.
             */
            std::shared_ptr<ESThtsCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;

            /**
             * Override the pretty print to print out the soft value, dp value, entropy and temperature
             */
            virtual std::string get_pretty_print_val() const;
        
        public:
            /**
             * Override create_child_node_helper_itfc still, so that the ThtsDNode create_child_node_itfc function can 
             * create the correct ESThtsCNode using the above version of create_child_node
             */
            virtual std::shared_ptr<ThtsCNode> create_child_node_helper_itfc(
                std::shared_ptr<const Action> action) const;
    };
}