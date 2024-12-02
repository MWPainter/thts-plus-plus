#pragma once

#include "mo/ch_thts_chance_node.h"

#include "mo/convex_hull.h"
#include "mo/ch_thts_manager.h"
#include "mo/mo_thts_decision_node.h"




namespace thts {
    // forward declare 
    class CH_MoThtsCNode;
    class CH_MoThtsManager;
    class MoThtsContext;

    /**
     * Base class for decision nodes that use ConvexHull objects for their state
    */
    class CH_MoThtsDNode : public MoThtsDNode {
        friend CH_MoThtsCNode;

        protected:
            int num_backups;
            ConvexHull<std::shared_ptr<const Action>> convex_hull;

        public:
            CH_MoThtsDNode(
                std::shared_ptr<CH_MoThtsManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const CH_MoThtsCNode> parent=nullptr); 

            virtual ~CH_MoThtsDNode() = default;
            
            virtual void visit(MoThtsContext& ctx);
            virtual std::shared_ptr<const Action> select_action(MoThtsContext& ctx) = 0;
            virtual std::shared_ptr<const Action> recommend_action(MoThtsContext& ctx) const;
            std::shared_ptr<const Action> recommend_action_from_tags(MoThtsContext& ctx) const;
            virtual void backup(
                const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
                const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
                const Eigen::ArrayXd trial_cumulative_return_after_node, 
                const Eigen::ArrayXd trial_cumulative_return,
                MoThtsContext& ctx);

            std::string get_convex_hull_pretty_print_string() const;

        protected:
            virtual std::shared_ptr<CH_MoThtsCNode> create_child_node_helper(
                std::shared_ptr<const Action> action) const = 0;
            virtual std::string get_pretty_print_val() const override = 0;
        


        /**
         * Boilerplate function definitions. 
         * 
         * Functionality implemented in thts_decision_node.h, but it's useful to have wrappers to avoid needing to 
         * use pointer casts frequently.
         * 
         * Boilerplate implementations provided in thts_decision_node_template.h
         */
        public:
            std::shared_ptr<CH_MoThtsCNode> create_child_node(std::shared_ptr<const Action> action);
            std::shared_ptr<CH_MoThtsCNode> get_child_node(std::shared_ptr<const Action> action) const;



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