#pragma once

#include "algorithms/rents_chance_node.h"
#include "algorithms/ments_manager.h"
#include "thts_types.h"

#include "ments_decision_node.h"
#include "thts_env.h"
#include "thts_env_context.h"
#include "thts_manager.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace thts {
    // forward declare corresponding RentsCNode class
    class RentsCNode;

    /**
     * An implementation of RENTS in the Thts schema
     * 
     * Paper: http://proceedings.mlr.press/v139/dam21a/dam21a.pdf
     * 
     * To pass the parent distributions (the distribution used by the previous DNode) in a setting running many 
     * concurrent trials, we use the ThteEnvContext.
     * 
     * Note that this implementation wont work if any node has zero common actions with its parent node. (And its 
     * questionable if RENTS may be useful in that case?)
     * 
     * Member variables:
     *      _node_distr_key: A key to use for storing this node's select action distribution in context's.
     *      _parent_distr_key: A key to use for accessing the parent's distribution from context's.
     */
    class RentsDNode : public MentsDNode {
        // Allow RentsCNode access to private members
        friend RentsCNode;

        /**
         * Core RentsDNode implementation.
         */
        protected:
            std::string _node_distr_key;
            std::string _parent_distr_key;

            /**
             * Gets the action distribution for a parent node 
             * 
             * Args:
             *      ctx: A thts env context containing the distribution of the parent node
             * 
             * Returns:
             *      The action distribution the parent node used, or, nullptr if this node has no parent decision node 
             *      (i.e. it is the root node/top level node)
            */
            std::shared_ptr<ActionDistr> get_parent_distr_from_context(ThtsEnvContext& ctx) const;

            /**
             * Puts the action distribution for this node into the thts env context
             * 
             * Args:
             *      action_distr: The distribution over actions computed in the select action phase to be stored
             *      ctx: A thts env context to store the distribution
            */
           void put_node_distr_in_context(std::shared_ptr<ActionDistr> action_distr, ThtsEnvContext& ctx) const;

           /**
            * Get prob from parent distribution (handling boundary cases at the root node)
            * 
            * For the boundary case at. theroot node, we always return a probability of 1.0, so that the normal 
            * ments distribution can be computed.
            * 
            * Args:
            *       parent_distr: The parent distribtion over actions (possibly nullptr)
            *       action: The action to get a probability for 
            * 
            * Returns:
            *       The probability that the parent distribution would sample 'action' (or a const 1.0 in the boundary 
            *       root node case)
           */
          double get_parent_action_prob(
            std::shared_ptr<ActionDistr> parent_distr, std::shared_ptr<const Action> action) const;

            /**
             * Computes the weights for each action.
             * 
             * (This excludes any probability mass from epsilon exploration).
             * 
             * Assumes that we already hold locks for all of the children.
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
             * Implements select_action for rents.
             * 
             * Computes the probability distribution to select an action from, stores it in the context, and returns 
             * the sampled action.
             * 
             * Args:
             *      ctx: A thts env context
             * 
             * Returns:
             *      The action selected.
             */
            std::shared_ptr<const Action> select_action_rents(ThtsEnvContext& ctx);





        /**
         * Core ThtsDNode implementation functions.
         */
        public:  
            /**
             * Constructor
             */
            RentsDNode(
                std::shared_ptr<MentsManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const RentsCNode> parent=nullptr); 

            virtual ~RentsDNode() = default;
            
            /**
             * Calls the rents select action method
             * 
             * Args:
             *      ctx: A context provided to all thts functions throughout a trial to pass intermediate/transient info
             * 
             * Returns:
             *      The selected action
             */
            virtual std::shared_ptr<const Action> select_action(ThtsEnvContext& ctx);

        protected:
            /**
             * A helper function that makes a child node object on the heap and returns it. 
             * 
             * The 'create_child_node' boilerplate function uses this function to make a new child, add it to the 
             * children map (or bypass making the node using the transposition table if using). The function is marked 
             * const to enforce that we don't accidently try to duplicate logic surrounding adding children and 
             * interacting with the transposition table.
             * 
             * Args:
             *      action: An action to create a child node for
             * 
             * Returns:
             *      A pointer to a new MentsCNode object
             */
            std::shared_ptr<RentsCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;



        /**
         * ThtsDNode interface function definitions, used by thts subroutines to interact with this node. Copied from 
         * thts_decision_node.h. 
         * 
         * Boilerplate definitions are provided in thts_decision_node_template.h, that wrap above functions in pointer 
         * casts.
         */
        public:
            virtual std::shared_ptr<ThtsCNode> create_child_node_helper_itfc(
                std::shared_ptr<const Action> action) const;
    };
}
