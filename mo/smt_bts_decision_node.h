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
     * SM-BTS impl
    */
    class SmtBtsDNode : public SmtThtsDNode {
        friend SmtBtsCNode;

        protected:
            int num_backups;

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

            /**
             * BTS code - get search temp
             */
            virtual double get_temp() const;

            /**
             * BTS code - Helper to get the q-value of an action. 
             * 
             * TODO: cleaner entropy interface
             * 
             * Args:
             *      action: 
             *          The action to get the corresponding q value for
             *      opponent_coeff: 
             *          A value of -1.0 or 1.0 for if we are acting as the opponent in a two player game or not 
             *          respectively
             *      q_val_map:
             *          A map of q values to be filled
             */
            virtual Eigen::ArrayXd get_q_value(
                std::shared_ptr<const Action> action, 
                double opponent_coeff, 
                MoThtsContext& ctx,
                double& entropy,
                bool& pure_backup_val) const;
            void get_child_q_values(
                ActionVector& actions,
                std::unordered_map<std::shared_ptr<const Action>,Eigen::ArrayXd>& q_val_map, 
                std::unordered_map<std::shared_ptr<const Action>,double>& entropy_map, 
                std::unordered_map<std::shared_ptr<const Action>,bool>& pure_backup_val_map,
                MoThtsContext& ctx) const;

            /**
             * BTS code - computes the weights for each action.
             * 
             * Args:
             *      q_val_map:
             *          The q 
             *      action_weights: 
             *          An ActionDistr to be filled with values of the form exp(q_value/temp - C), where C is equal to
             *          max(q_value/temp)
             *      normalisation_term:
             *          A double reference to be filled with the value of C from 'action_weights' description.
             *      context:
             *          A thts env context
             */
            virtual void compute_action_weights(
                ActionVector& actions,
                std::unordered_map<std::shared_ptr<const Action>,Eigen::ArrayXd>& q_val_map,
                std::unordered_map<std::shared_ptr<const Action>,double>& entropy_map, 
                ActionDistr& action_weights, 
                double& sum_action_weights, 
                double& normalisation_term, 
                MoThtsContext& context) const;

            /**
             * BTS code - computes the action distribution
             * 
             * TODO: add ability for prior policy here at later date
             * 
             * Args:
             *      action_distr:
             *          An ActionDistr to be filled with a normalised probability distribution to select actions with
             *      context:
             *          A thts env context
             */
            void compute_action_distribution(
                ActionVector& actions,
                std::unordered_map<std::shared_ptr<const Action>,Eigen::ArrayXd>& q_val_map,
                std::unordered_map<std::shared_ptr<const Action>,double>& entropy_map, 
                ActionDistr& action_distr, 
                MoThtsContext& context) const;

            std::string get_simplex_map_pretty_print_string() const;

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