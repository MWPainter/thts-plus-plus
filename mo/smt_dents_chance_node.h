#pragma once

#include "mo/smt_bts_chance_node.h"
#include "mo/smt_dents_manager.h"
#include "mo/smt_dents_decision_node.h"




namespace thts {
    // forward declare 
    class SmtDentsDNode;
    class SmtDentsManager;
    class MoThtsContext;

    /**
     * SM-DENTS impl
    */
    class SmtDentsCNode : public SmtBtsCNode {
        friend SmtDentsDNode;

        public:
            SmtDentsCNode(
                std::shared_ptr<SmtDentsManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const SmtDentsDNode> parent=nullptr); 

            virtual ~SmtDentsCNode() = default;
            
            virtual void backup(
                const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
                const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
                const Eigen::ArrayXd trial_cumulative_return_after_node, 
                const Eigen::ArrayXd trial_cumulative_return,
                MoThtsContext& ctx) override;

        protected:
            virtual std::shared_ptr<SmtThtsDNode> create_child_node_helper(
                std::shared_ptr<const State> next_state) const override;
    };
}