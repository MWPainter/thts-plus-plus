#pragma once

#include "mo/bl_thts_chance_node.h"

#include "mo/ball_list.h"
#include "mo/bl_thts_manager.h"
#include "mo/mo_thts_chance_node.h"




namespace thts {
    // forward declare 
    class BL_MoThtsDNode;
    class MoThtsContext;

    /**
     * Base class for decision nodes that use CZ_BallList objects for their state
    */
    class BL_MoThtsCNode : public MoThtsCNode {
        friend BL_MoThtsDNode;

        protected:
            CZ_BallList ball_list;

        public:
            BL_MoThtsCNode(
                std::shared_ptr<BL_MoThtsManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const BL_MoThtsDNode> parent=nullptr); 

            virtual ~BL_MoThtsCNode() = default;
            
            virtual void visit(MoThtsContext& ctx);
            virtual std::shared_ptr<const State> sample_observation(MoThtsContext& ctx) = 0;
            virtual void backup(
                const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
                const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
                const Eigen::ArrayXd trial_cumulative_return_after_node, 
                const Eigen::ArrayXd trial_cumulative_return,
                MoThtsContext& ctx) = 0;

            std::string get_ball_list_pretty_print_string() const;

        protected:
            virtual std::shared_ptr<BL_MoThtsDNode> create_child_node_helper(
                std::shared_ptr<const State> state) const = 0;
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
            std::shared_ptr<BL_MoThtsDNode> create_child_node(std::shared_ptr<const State> next_state);
            std::shared_ptr<BL_MoThtsDNode> get_child_node(std::shared_ptr<const State> next_state) const;



        /**
         * ThtsDNode interface function definitions, used by thts subroutines to interact with this node. Copied from 
         * thts_decision_node.h. 
         * 
         * Boilerplate definitions are provided in thts_decision_node_template.h, that wrap above functions in pointer 
         * casts.
         */
        public:
            virtual void visit_itfc(ThtsEnvContext& ctx) override;
            virtual std::shared_ptr<const Observation> sample_observation_itfc(ThtsEnvContext& ctx) override;
            virtual void backup_itfc(
                const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
                const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
                const Eigen::ArrayXd trial_cumulative_return_after_node, 
                const Eigen::ArrayXd trial_cumulative_return,
                ThtsEnvContext& ctx) override;

            virtual std::shared_ptr<ThtsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation, 
                std::shared_ptr<const State> next_state=nullptr) const override;
    };
}