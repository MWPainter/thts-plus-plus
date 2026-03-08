#pragma once

#include "mo/algorithms/simplex_maps/sm_decision_node.h"
#include "mo/algorithms/simplex_maps/sm_bts_manager.h"
#include "mo/algorithms/simplex_maps/sm_bts_chance_node.h"





namespace thts {
    // forward declare 
    class SmBtsCNode;
    class SmBtsManager;
    class MoThtsContext;

    /**
     * SM-BTS impl
    */
    class SmBtsDNode : public SmThtsDNode {
        friend SmBtsCNode;

        protected:
            int num_backups;

        public:
            SmBtsDNode(
                std::shared_ptr<SmBtsManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const SmBtsCNode> parent=nullptr); 

            virtual ~SmBtsDNode() = default;
            
            // virtual void visit(MoThtsContext& ctx)  override;
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
            virtual double get_temp(MoThtsContext& ctx) const;

            /**
             * BTS code - Helper to get the q-value(s) from children
             Value estimates (for search), in both MO vector + scalar utility form
             And Entropy
             And how many visits the vertex in the child simplex map had (in case want to ignore num_updates==0 cases)
             
             Adds an option to only get values from children 
             And option to ignore children that give a value with num_updates == 0
             */
            void read_values_from_child_(
                std::shared_ptr<const Action> action, 
                Vec& weight,
                int& value_estimate_num_updates_,
                Vec& value_estimate_,
                Vec& value_estimate_local_,
                double& entropy_estimate_) const;
            void fill_child_values_maps_(
                ActionVector& actions,
                Vec& weight,
                std::unordered_map<std::shared_ptr<const Action>,int>& value_estimate_num_updates_map_,
                std::unordered_map<std::shared_ptr<const Action>,Vec>& value_estimate_map_,
                std::unordered_map<std::shared_ptr<const Action>,Vec>& value_estimate_local_map_,
                std::unordered_map<std::shared_ptr<const Action>,double>& entropy_estimate_map_) const;
            
            /**
            Helpers for manupulating the maps of values from children
             */
            std::unordered_map<std::shared_ptr<const Action>,double> utility_weights_from_values(
                Vec& weight,
                std::unordered_map<std::shared_ptr<const Action>,Vec>& values) const;

            /**
             * BTS code - computes the weights for each action. (weights for boltzmann distribution)
             */
            virtual void compute_action_weights_(
                ActionVector& actions,
                MoThtsContext& context,
                ActionDistr& action_weights_,
                double& sum_weights_) const;
            virtual void compute_action_weights_helper_(
                ActionVector& actions,
                MoThtsContext& context,
                std::unordered_map<std::shared_ptr<const Action>,Vec>& value_estimate_local_map,
                std::unordered_map<std::shared_ptr<const Action>,double>& entropy_estimate_map,
                ActionDistr& action_weights_,
                double& sum_weights_) const;

            /**
             * BTS code - computes the action distribution for BTS
             Basically mixes in eps greedy mass into the boltzmann distr from compute_action_weights_
             */
            virtual void compute_action_distribution_(
                ActionVector& actions,
                MoThtsContext& context,
                ActionDistr& action_distr_) const;
            virtual void compute_action_distribution_helper_(
                ActionVector& actions,
                MoThtsContext& context,
                ActionDistr& action_distr_, // helper where action_distr_ is already filled with boltzmann action weights
                double sum_weights) const;

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
            std::shared_ptr<SmBtsCNode> create_child_node(std::shared_ptr<const Action> action);
            virtual std::shared_ptr<SmThtsCNode> create_child_node_helper(
                std::shared_ptr<const Action> action) const override;
            std::shared_ptr<SmBtsCNode> get_child_node(std::shared_ptr<const Action> action) const;
    };
}