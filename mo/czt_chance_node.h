#pragma once

#include "mo/bl_thts_chance_node.h"
#include "mo/czt_manager.h"
#include "mo/czt_chance_node.h"




namespace thts {
    // forward declare 
    class CztDNode;
    class CztManager;
    class MoThtsContext;

    /**
     * CZT impl
    */
    class CztCNode : public BL_MoThtsCNode {
        friend CztDNode;

        public:
            CztCNode(
                std::shared_ptr<CztManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const CztDNode> parent=nullptr); 

            virtual ~CztCNode() = default;
            
            virtual void visit(MoThtsContext& ctx);
            virtual std::shared_ptr<const State> sample_observation(MoThtsContext& ctx) override;
            virtual void backup(
                const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
                const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
                const Eigen::ArrayXd trial_cumulative_return_after_node, 
                const Eigen::ArrayXd trial_cumulative_return,
                MoThtsContext& ctx) override;

        protected:
            virtual std::string get_pretty_print_val() const override;
            virtual std::shared_ptr<BL_MoThtsDNode> create_child_node_helper(
                std::shared_ptr<const State> next_state) const override;
        


        /**
         * Boilerplate function definitions. 
         * 
         * Functionality implemented in thts_decision_node.h, but it's useful to have wrappers to avoid needing to 
         * use pointer casts frequently.
         * 
         * Boilerplate implementations provided in thts_decision_node_template.h
         */
        public:
            std::shared_ptr<CztDNode> create_child_node(std::shared_ptr<const State> next_state);
            std::shared_ptr<CztDNode> get_child_node(std::shared_ptr<const State> next_state) const;
    };
}