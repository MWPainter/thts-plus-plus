#pragma once

#include "mo/smt_bts_decision_node.h"
#include "mo/smt_dents_manager.h"
#include "mo/smt_dents_chance_node.h"





namespace thts {
    // forward declare 
    class SmtDentsCNode;
    class SmtDentsManager;
    class MoThtsContext;

    /**
     * SM-DENTS impl
    */
    class SmtDentsDNode : public SmtBtsDNode {
        friend SmtDentsCNode;

        public:
            SmtDentsDNode(
                std::shared_ptr<SmtDentsManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const SmtDentsCNode> parent=nullptr); 

            virtual ~SmtDentsDNode() = default;
            
            virtual void backup(
                const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
                const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
                const Eigen::ArrayXd trial_cumulative_return_after_node, 
                const Eigen::ArrayXd trial_cumulative_return,
                MoThtsContext& ctx)  override;

            /**
             * BTS code - get value temp
             */
            virtual double get_value_temp() const;

            /**
             * BTS code - computes the weights for each action.
             * 
             * Args:
             *      q_val_map:
             *          The q 
             *      action_weights: 
             *          An ActionDistr to be filled with values of the form exp(q_value/temp - C), where C is equal to
             *          max(q_value/temp)
             *      normalisation_term:
             *          A double reference to be filled with the value of C from 'action_weights' description.
             *      context:
             *          A thts env context
             */
            virtual void compute_action_weights(
                ActionVector& actions,
                std::unordered_map<std::shared_ptr<const Action>,Eigen::ArrayXd>& q_val_map,
                std::unordered_map<std::shared_ptr<const Action>,double>& entropy_map, 
                ActionDistr& action_weights, 
                double& sum_action_weights, 
                double& normalisation_term, 
                MoThtsContext& context) const;
        


        /**
         * Boilerplate function definitions. 
         * 
         * Functionality implemented in thts_decision_node.h, but it's useful to have wrappers to avoid needing to 
         * use pointer casts frequently.
         * 
         * Boilerplate implementations provided in thts_decision_node_template.h
         */
        public:
            virtual std::shared_ptr<SmtThtsCNode> create_child_node_helper(
                std::shared_ptr<const Action> action) const override;
    };
}