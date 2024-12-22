#pragma once

#include "mo/mo_thts_chance_node.h"

#include "mo/simplex_map.h"
#include "mo/smt_manager.h"
#include "mo/smt_decision_node.h"




namespace thts {
    // forward declare 
    class SmtThtsDNode;
    class MoThtsContext;

    /**
     * Base class for decision nodes that use SimplexMap objects for their state
    */
    class SmtThtsCNode : public MoThtsCNode {
        friend SmtThtsDNode;

        // TODO: change this back to protected after debugging stuff
        // protected:
        public:
            SimplexMap simplex_map;

        public:
            SmtThtsCNode(
                std::shared_ptr<SmtThtsManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const SmtThtsDNode> parent=nullptr); 

            virtual ~SmtThtsCNode() = default;
            
            virtual void visit(MoThtsContext& ctx);
            virtual std::shared_ptr<const State> sample_observation(MoThtsContext& ctx) = 0;
            virtual void backup(
                const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
                const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
                const Eigen::ArrayXd trial_cumulative_return_after_node, 
                const Eigen::ArrayXd trial_cumulative_return,
                MoThtsContext& ctx) = 0;

            std::string get_simplex_map_pretty_print_string() const;

        protected:
            virtual std::shared_ptr<SmtThtsDNode> create_child_node_helper(
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
            std::shared_ptr<SmtThtsDNode> create_child_node(std::shared_ptr<const State> next_state);
            std::shared_ptr<SmtThtsDNode> get_child_node(std::shared_ptr<const State> next_state) const;



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