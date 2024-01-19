#pragma once

#include "mo/bl_thts_decision_node.h"
#include "mo/czt_manager.h"
#include "mo/czt_chance_node.h"





namespace thts {
    // forward declare 
    class CztCNode;
    class CztManager;
    class MoThtsContext;

    /**
     * CZT impl
    */
    class CztDNode : public BL_MoThtsDNode {
        friend CztCNode;

        private:
            std::string _action_ctx_key;
            std::string _ball_ctx_key;

        public:
            CztDNode(
                std::shared_ptr<CztManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const CztCNode> parent=nullptr); 

            virtual ~CztDNode() = default;
            
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

        private:
            void fill_cz_values_and_ball_ptrs(
                ActionVector& actions,
                std::unordered_map<std::shared_ptr<const Action>,double>& cz_values, 
                std::unordered_map<std::shared_ptr<const Action>,std::shared_ptr<CZ_Ball>>& cz_balls, 
                MoThtsContext& ctx);
        


        /**
         * Boilerplate function definitions. 
         * 
         * Functionality implemented in thts_decision_node.h, but it's useful to have wrappers to avoid needing to 
         * use pointer casts frequently.
         * 
         * Boilerplate implementations provided in thts_decision_node_template.h
         */
        public:
            std::shared_ptr<CztCNode> create_child_node(std::shared_ptr<const Action> action);
            virtual std::shared_ptr<BL_MoThtsCNode> create_child_node_helper(
                std::shared_ptr<const Action> action) const override;
            std::shared_ptr<CztCNode> get_child_node(std::shared_ptr<const Action> action) const;
    };
}