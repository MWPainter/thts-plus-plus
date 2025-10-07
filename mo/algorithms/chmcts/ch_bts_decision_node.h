#pragma once

#include "mo/algorithms/chmcts/ch_bts_chance_node.h"

#include "mo/algorithms/chmcts/ch_bts_manager.h"
#include "mo/algorithms/chmcts/ch_thts_decision_node.h"




namespace thts {
    // forward declare 
    class ChBtsCNode;
    class ChBtsManager;
    class MoThtsContext;

    /**
     * CHMCTS Decision node
     * 
     * This code is quite messy, but don't plan to support it long term, sorry if you're reading this
    */
    class ChBtsDNode : public ChThtsDNode {
        friend ChBtsCNode;

        protected:

        public:
            ChBtsDNode(
                std::shared_ptr<ChBtsManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const ChBtsCNode> parent=nullptr); 

            virtual ~ChBtsDNode() = default;
            
            // virtual void visit(MoThtsContext& ctx) override;
            virtual std::shared_ptr<const Action> select_action(MoThtsContext& ctx) override;
            // virtual std::shared_ptr<const Action> recommend_action(MoThtsContext& ctx) const override;
            // virtual void backup(
            //     const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
            //     const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
            //     const Eigen::ArrayXd trial_cumulative_return_after_node, 
            //     const Eigen::ArrayXd trial_cumulative_return,
            //     MoThtsContext& ctx) override;

        protected:
            virtual std::string get_pretty_print_val() const override;
            
            /**
             * Copied from ments decision node
             */
            virtual double get_temp(MoThtsContext& context) const;
            virtual void compute_action_weights(
                ActionDistr& action_weights, 
                double& sum_action_weights, 
                double& numerical_stability_term, 
                MoThtsContext& context) const;
            void compute_action_distribution(
                ActionDistr& action_distr, 
                MoThtsContext& context) const;
        


        /**
         * Boilerplate function definitions. 
         * 
         * Functionality implemented in thts_decision_node.h, but it's useful to have wrappers to avoid needing to 
         * use pointer casts frequently.
         * 
         * Boilerplate implementations provided in thts_decision_node_template.h
         */
        public:
            std::shared_ptr<ChBtsCNode> create_child_node(std::shared_ptr<const Action> action);
            virtual std::shared_ptr<ChThtsCNode> create_child_node_helper(
                std::shared_ptr<const Action> action) const override;
            std::shared_ptr<ChBtsCNode> get_child_node(std::shared_ptr<const Action> action) const;



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