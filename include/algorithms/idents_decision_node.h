#pragma once

#include "algorithms/idents_chance_node.h"
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
    // forward declare corresponding IDentsCNode class
    class IDentsCNode;

    /**
     * An implementation of I(mproved)DENTS in the Thts schema
     * 
     * This version of dents runs ments, but the values that it uses to recommend and sample from of the form 
     * V_soft - (compute_temp - decayed_temp) * H, where H is the policy entropy of the subtree rooted at this node, 
     * compute temp is the temperature that ments used, and decayed temp is the desired temperature to use. For two 
     * player games, we replace the H term with (H_player - H_opponent), to encorporate the player and opponent's 
     * entropies contributions.
     * 
     * This node now needs to keep estimates of the local and sub-tree entropy, and use it in the computation of 
     * probability weights.
     * 
     * Attributes:
     *      local_entropy: The entropy of the current policy at this node
     *      subtree_entropy: The entropy of the policy for the entire subtree
     */
    class IDentsDNode : public MentsDNode {
        // Allow IDentsCNode access to private members
        friend IDentsCNode;

        protected: 
            double ments_local_entropy;
            double ments_subtree_entropy;
            double local_entropy;
            double subtree_entropy;

        public:  
            /**
             * Constructor
             */
            IDentsDNode(
                std::shared_ptr<IDentsManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const IDentsCNode> parent=nullptr); 

            virtual ~IDentsDNode() = default;

            /**
             * Helper to get the temperature that should be used in ments functions.
             */
            virtual double get_temp() const;

            /**
             * Helper to get the value of the decayed temperature to use. (I.e. the actual coefficient that we want 
             * of the entropy)
             */
            virtual double get_decayed_temp() const;

            /**
             * TODO
            */
            virtual double get_decayed_soft_q_value(std::shared_ptr<const Action> action, double opponent_coeff) const;

            /**
             * Computes the weights for each action.
             * 
             * (This excludes any probability mass from epsilon exploration).
             * 
             * Assumes that we already hold locks for all of the children.
             * 
             * TODO: would be nice to seperate the repeated logic out, but would require restructuring the ments classes
             * 
             * Args:
             *      action_weights: 
             *          An ActionDistr to be filled with values of the form exp(q_value/temp - C), where C is equal to
             *          max(q_value/temp)
             *      sun_action_weights:
             *          A double reference to be filled with the sum of all the weights in 'action_weights'
             *      normalisation_term:
             *          A double reference to be filled with the value of C from 'action_weights' description.
             *      context:
             *          A thts env context
             */
            virtual void compute_action_weights(
                ActionDistr& action_weights, 
                double& sum_action_weights, 
                double& normalisation_term, 
                ThtsEnvContext& context) const;

            /**
             * TODO
             * 
             * Urghghghghh code ugly
            */
            void compute_ments_action_distribution(
                ActionDistr& action_distr, 
                ThtsEnvContext& context) const;

            /**
             * Implements a soft backup for idents.
             * 
             * This is an edited version of the backup for ments. It additionally computes the values of the entropy 
             * variables.
             * 
             * Is thread safe, and will lock children before trying to access them.
             * 
             * TODO: would be nice to seperate the repeated logic out, but would require restructuring the ments classes
             * 
             * Args:
             *      ctx: A thts env context
             */
            void backup_soft(ThtsEnvContext& ctx);

            /**
             * TODO
            */
            void backup_entropy(ThtsEnvContext& ctx);
            
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
             * Helper to make a IDentsCNode child object.
             */
            std::shared_ptr<IDentsCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;

            /**
             * Override the pretty print to print out the soft value and the temp for debugging
             */
            virtual std::string get_pretty_print_val() const;
        
        public:
            /**
             * Override create_child_node_helper_itfc still, so that the ThtsDNode create_child_node_itfc function can 
             * create the correct IDentsCNode using the above version of create_child_node
             */
            virtual std::shared_ptr<ThtsCNode> create_child_node_helper_itfc(
                std::shared_ptr<const Action> action) const;
    };
}
