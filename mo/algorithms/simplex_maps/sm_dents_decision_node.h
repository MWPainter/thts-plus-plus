#pragma once

#include "mo/algorithms/simplex_maps/sm_bts_decision_node.h"
#include "mo/algorithms/simplex_maps/sm_dents_manager.h"
#include "mo/algorithms/simplex_maps/sm_dents_chance_node.h"





namespace thts {
    // forward declare 
    class SmDentsCNode;
    class SmDentsManager;
    class MoThtsContext;

    /**
     * SM-DENTS impl
    */
    class SmDentsDNode : public SmBtsDNode {
        friend SmDentsCNode;

        public:
            SmDentsDNode(
                std::shared_ptr<SmDentsManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const SmDentsCNode> parent=nullptr); 

            virtual ~SmDentsDNode() = default;
            
            virtual void backup(
                const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
                const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
                const Eigen::ArrayXd trial_cumulative_return_after_node, 
                const Eigen::ArrayXd trial_cumulative_return,
                MoThtsContext& ctx)  override;

            /**
             * DENTS code - get value temp
             */
            virtual double get_entropy_coeff(MoThtsContext& ctx) const;

            /**
             * Action distribution overrides
             */
            virtual void compute_action_weights_helper_(
                ActionVector& actions,
                MoThtsContext& context,
                std::unordered_map<std::shared_ptr<const Action>,Vec>& value_estimate_for_search_map,
                std::unordered_map<std::shared_ptr<const Action>,double>& entropy_estimate_map,
                ActionDistr& action_weights_,
                double& sum_weights_) const override;

        /**
         * Boilerplate function definitions. 
         * 
         * Functionality implemented in thts_decision_node.h, but it's useful to have wrappers to avoid needing to 
         * use pointer casts frequently.
         * 
         * Boilerplate implementations provided in thts_decision_node_template.h
         */
        public:
            virtual std::shared_ptr<SmThtsCNode> create_child_node_helper(
                std::shared_ptr<const Action> action) const override;
    };
}