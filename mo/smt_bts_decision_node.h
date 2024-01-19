#pragma once

#include "mo/smt_decision_node.h"
#include "mo/smt_bts_manager.h"
#include "mo/smt_bts_chance_node.h"





namespace thts {
    // forward declare 
    class SmtBtsCNode;
    class SmtBtsManager;
    class MoThtsContext;

    /**
     * CZT impl
    */
    class SmtBtsDNode : public SmtThtsDNode {
        friend SmtBtsCNode;

        private:
            std::string _action_ctx_key;
            std::string _ball_ctx_key;

        public:
            SmtBtsDNode(
                std::shared_ptr<SmtBtsManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const SmtBtsCNode> parent=nullptr); 

            virtual ~SmtBtsDNode() = default;
            
            virtual void visit(MoThtsContext& ctx)  override;
            virtual std::shared_ptr<const Action> select_action(MoThtsContext& ctx)  override;
            virtual std::shared_ptr<const Action> recommend_action(MoThtsContext& ctx) const  override;
            virtual void backup(
                const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
                const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
                const Eigen::ArrayXd trial_cumulative_return_after_node, 
                const Eigen::ArrayXd trial_cumulative_return,
                MoThtsContext& ctx)  override;

        protected:
            virtual std::string get_pretty_print_val() const override;
        


        /**
         * Boilerplate function definitions. 
         * 
         * Functionality implemented in thts_decision_node.h, but it's useful to have wrappers to avoid needing to 
         * use pointer casts frequently.
         * 
         * Boilerplate implementations provided in thts_decision_node_template.h
         */
        public:
            std::shared_ptr<SmtBtsCNode> create_child_node(std::shared_ptr<const Action> action);
            virtual std::shared_ptr<SmtThtsCNode> create_child_node_helper(
                std::shared_ptr<const Action> action) const override;
            std::shared_ptr<SmtBtsCNode> get_child_node(std::shared_ptr<const Action> action) const;
    };
}