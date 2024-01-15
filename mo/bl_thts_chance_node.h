#pragma once

#include "mo/mo_thts_chance_node.h"




namespace thts {
    // forward declare 
    class BL_MoThtsDNode;

    /**
     * Base class for decision nodes that use CZ_BallList objects for their state
    */
    class BL_MoThtsCNode : public MoThtsCNode {

        protected:
            CZ_BallList ball_list;

        public:
            BL_MoThtsCNode(
                std::shared_ptr<MoThtsManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const BL_MoThtsDNode> parent=nullptr); 

            virtual ~BL_MoThtsCNode() = default;
            
            virtual void visit(MoThtsEnvContext& ctx);
            virtual std::shared_ptr<const State> sample_observation(MoThtsEnvContext& ctx) = 0;
            virtual void backup(
                const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
                const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
                const Eigen::ArrayXd trial_cumulative_return_after_node, 
                const Eigen::ArrayXd trial_cumulative_return,
                MoThtsEnvContext& ctx) = 0;

        protected:
            // std::shared_ptr<BL_MoThtsDNode> create_child_node_helper(std::shared_ptr<const State> obs) const;
            // virtual std::string get_pretty_print_val() const override;
        


        /**
         * Boilerplate function definitions. 
         * 
         * Functionality implemented in thts_decision_node.h, but it's useful to have wrappers to avoid needing to 
         * use pointer casts frequently.
         * 
         * Boilerplate implementations provided in thts_decision_node_template.h
         */
        public:
            std::shared_ptr<BL_MoThtsDNode> create_child_node(std::shared_ptr<const State> obs);
            std::shared_ptr<BL_MoThtsDNode> get_child_node(std::shared_ptr<const State> obs) const;



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
                std::shared_ptr<const Observation> observation, std::shared_ptr<const State> next_state=nullptr) const;
    }
}