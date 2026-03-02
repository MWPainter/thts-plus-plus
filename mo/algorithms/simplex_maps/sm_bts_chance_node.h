#pragma once

#include "mo/algorithms/simplex_maps/sm_chance_node.h"
#include "mo/algorithms/simplex_maps/sm_bts_manager.h"
#include "mo/algorithms/simplex_maps/sm_bts_decision_node.h"




namespace thts {
    // forward declare 
    class SmBtsDNode;
    class SmBtsManager;
    class MoThtsContext;

    /**
     * SM-BTS impl
    */
    class SmBtsCNode : public SmThtsCNode {
        friend SmBtsDNode;

        protected:
            int num_backups;
            Vec local_reward;

        public:
            SmBtsCNode(
                std::shared_ptr<SmBtsManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const SmBtsDNode> parent=nullptr); 

            virtual ~SmBtsCNode() = default;
            
            // virtual void visit(MoThtsContext& ctx);
            virtual std::shared_ptr<const State> sample_observation(MoThtsContext& ctx) override;
            virtual void backup(
                const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
                const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
                const Eigen::ArrayXd trial_cumulative_return_after_node, 
                const Eigen::ArrayXd trial_cumulative_return,
                MoThtsContext& ctx) override;

            std::string get_simplex_map_pretty_print_string() const;

        protected:
            virtual std::string get_pretty_print_val() const override;
            virtual std::shared_ptr<SmThtsDNode> create_child_node_helper(
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
            std::shared_ptr<SmBtsDNode> create_child_node(std::shared_ptr<const State> next_state);
            std::shared_ptr<SmBtsDNode> get_child_node(std::shared_ptr<const State> next_state) const;
    };
}