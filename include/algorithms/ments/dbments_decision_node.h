#pragma once

#include "algorithms/ments/dbments_chance_node.h"
#include "algorithms/ments/ments_manager.h"
#include "thts_types.h"

#include "algorithms/ments/ments_chance_node.h"
#include "algorithms/ments/ments_decision_node.h"
#include "algorithms/common/dp_chance_node.h"
#include "algorithms/common/dp_decision_node.h"
#include "thts_env.h"
#include "thts_env_context.h"
#include "thts_manager.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace thts {
    // forward declare corresponding DBMentsCNode class
    class DBMentsCNode;
    class DBMentsLogger;

    /**
     * An implementation of DB-MENTS in the Thts schema
     * 
     * The implementation only needs to change recommend action and backup functions to use dp recommend/backups.
     */
    class DBMentsDNode : public MentsDNode, public DPDNode {
        // Allow DBMentsCNode access to private members
        friend DBMentsCNode;
        friend DBMentsLogger;

        protected: 

        public:  
            /**
             * Constructor
             */
            DBMentsDNode(
                std::shared_ptr<MentsManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const DBMentsCNode> parent=nullptr); 

            virtual ~DBMentsDNode() = default;
            
            /**
             * Visit calls mentsdnode's and dpdnode's visit functions
             * 
             * Args:
             *      ctx: A context provided to all thts functions throughout a trial to pass intermediate/transient info
             */
            virtual void visit(ThtsEnvContext& ctx);

            // Hacking in avg returns recommend for go experiments
            std::shared_ptr<const Action> recommend_action_best_avg_return() const;
            
            /**
             * Uses the DPDNode to recommend an action according to the DP values.
             * 
             * Recommends a random action if this node has zero children.
             */
            virtual std::shared_ptr<const Action> recommend_action_best_dp_value() const;
            
            /**
             * Implements the thts recommend_action function for the node
             * 
             * Args:
             *      ctx: A context for if a recommendation also requires a context
             * 
             * Returns:
             *      The recommended action
             */
            virtual std::shared_ptr<const Action> recommend_action(ThtsEnvContext& ctx) const;
            
            /**
             * Calls both the soft backup from MentsDNode and dp backup from DPDNode
             */
            virtual void backup(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                ThtsEnvContext& ctx);

        protected:
            /**
             * Helper to make a DBMentsCNode child object.
             */
            std::shared_ptr<DBMentsCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;

            /**
             * Override the pretty print to print out both the dp value and the soft value
             */
            virtual std::string get_pretty_print_val() const;
        
        public:
            /**
             * Override create_child_node_helper_itfc still, so that the ThtsDNode create_child_node_itfc function can 
             * create the correct DBMentsCNode using the above version of create_child_node
             */
            virtual std::shared_ptr<ThtsCNode> create_child_node_helper_itfc(
                std::shared_ptr<const Action> action) const;
    };
}