#pragma once

#include "mo/chmcts_chance_node.h"

#include "mo/chmcts_manager.h"
#include "mo/ch_thts_decision_node.h"
#include "mo/czt_decision_node.h"




namespace thts {
    // forward declare 
    class ChmctsCNode;
    class ChmctsManager;
    class MoThtsContext;

    /**
     * CHMCTS Decision node
    */
    class ChmctsDNode : public CH_MoThtsDNode, public CztDNode {
        friend ChmctsCNode;

        public:
            ChmctsDNode(
                std::shared_ptr<ChmctsManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const ChmctsCNode> parent=nullptr); 

            virtual ~ChmctsDNode() = default;
            
            virtual void visit(MoThtsContext& ctx) override;
            virtual std::shared_ptr<const Action> select_action(MoThtsContext& ctx) override;
            virtual std::shared_ptr<const Action> recommend_action(MoThtsContext& ctx) const override;
            virtual void backup(
                const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
                const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
                const Eigen::ArrayXd trial_cumulative_return_after_node, 
                const Eigen::ArrayXd trial_cumulative_return,
                MoThtsContext& ctx) override;

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
            std::shared_ptr<ChmctsCNode> create_child_node(std::shared_ptr<const Action> action);
            virtual std::shared_ptr<ChmctsCNode> create_child_node_helper(
                std::shared_ptr<const Action> action) const override;
            std::shared_ptr<ChmctsCNode> get_child_node(std::shared_ptr<const Action> action) const;



        /**
         * ThtsDNode interface function definitions, used by thts subroutines to interact with this node. Copied from 
         * thts_decision_node.h. 
         * 
         * Boilerplate definitions are provided in thts_decision_node_template.h, that wrap above functions in pointer 
         * casts.
         */
        public:
            virtual void visit_itfc(ThtsEnvContext& ctx) override;
            virtual std::shared_ptr<const Action> select_action_itfc(ThtsEnvContext& ctx) override;
            virtual std::shared_ptr<const Action> recommend_action_itfc(ThtsEnvContext& ctx) const override;
            virtual void backup_itfc(
                const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
                const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
                const Eigen::ArrayXd trial_cumulative_return_after_node, 
                const Eigen::ArrayXd trial_cumulative_return,
                ThtsEnvContext& ctx) override;

            virtual std::shared_ptr<ThtsCNode> create_child_node_helper_itfc(
                std::shared_ptr<const Action> action) const override;
    };
}