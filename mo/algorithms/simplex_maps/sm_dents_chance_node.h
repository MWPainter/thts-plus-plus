#pragma once

#include "mo/algorithms/simplex_maps/sm_bts_chance_node.h"
#include "mo/algorithms/simplex_maps/sm_dents_manager.h"
#include "mo/algorithms/simplex_maps/sm_dents_decision_node.h"




namespace thts {
    // forward declare 
    class SmDentsDNode;
    class SmDentsManager;
    class MoThtsContext;

    /**
     * SM-DENTS impl
    */
    class SmDentsCNode : public SmBtsCNode {
        friend SmDentsDNode;

        public:
            SmDentsCNode(
                std::shared_ptr<SmDentsManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const SmDentsDNode> parent=nullptr); 

            virtual ~SmDentsCNode() = default;
            
            virtual void backup(
                const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
                const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
                const Eigen::ArrayXd trial_cumulative_return_after_node, 
                const Eigen::ArrayXd trial_cumulative_return,
                MoThtsContext& ctx) override;

        protected:
            virtual std::shared_ptr<SmThtsDNode> create_child_node_helper(
                std::shared_ptr<const State> next_state) const override;
    };
}