#pragma once

#include "mo/algorithms/contextual_zooming/bl_thts_chance_node.h"

#include "mo/data_structures/ball_list.h"
#include "mo/algorithms/contextual_zooming/bl_thts_manager.h"
#include "mo/mo_thts_decision_node.h"




namespace thts {
    // forward declare 
    class BlThtsCNode;
    class BlThtsManager;
    class MoThtsContext;

    /**
     * Base class for decision nodes that use CzBallList objects for their state
    */
    class BlThtsDNode : public MoThtsDNode {
        friend BlThtsCNode;

        // TODO: change this back to protected after debugging stuff
        // protected:
        public:
            CzBallList ball_list;

        public:
            BlThtsDNode(
                std::shared_ptr<BlThtsManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const BlThtsCNode> parent=nullptr,
                bool eval_mo_heuristic=true); 

            virtual ~BlThtsDNode() = default;
            
            virtual void visit(MoThtsContext& ctx);
            virtual std::shared_ptr<const Action> select_action(MoThtsContext& ctx) = 0;
            virtual std::shared_ptr<const Action> recommend_action(MoThtsContext& ctx) const = 0;
            virtual void backup(
                const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
                const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
                const Eigen::ArrayXd trial_cumulative_return_after_node, 
                const Eigen::ArrayXd trial_cumulative_return,
                MoThtsContext& ctx) = 0;

            std::string get_ball_list_pretty_print_string() const;

        protected:
            virtual std::shared_ptr<BlThtsCNode> create_child_node_helper(
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
            std::shared_ptr<BlThtsCNode> create_child_node(std::shared_ptr<const Action> action);
            std::shared_ptr<BlThtsCNode> get_child_node(std::shared_ptr<const Action> action) const;



        /**
         * ThtsDNode interface function definitions, used by thts subroutines to interact with this node. Copied from 
         * thts_decision_node.h. 
         * 
         * Boilerplate definitions are provided in thts_decision_node_template.h, that wrap above functions in pointer 
         * casts.
         */
        public:
            virtual void visit_itfc(ThtsContext& ctx) override;
            virtual std::shared_ptr<const Action> select_action_itfc(ThtsContext& ctx) override;
            virtual std::shared_ptr<const Action> recommend_action_itfc(ThtsContext& ctx) const override;
            virtual void backup_itfc(
                const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
                const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
                const Eigen::ArrayXd trial_cumulative_return_after_node, 
                const Eigen::ArrayXd trial_cumulative_return,
                ThtsContext& ctx) override;

            virtual std::shared_ptr<ThtsCNode> create_child_node_helper_itfc(
                std::shared_ptr<const Action> action) const override;
    };
}