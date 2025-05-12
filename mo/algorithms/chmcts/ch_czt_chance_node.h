#pragma once

#include "mo/algorithms/chmcts/ch_czt_decision_node.h"

#include "mo/algorithms/chmcts/ch_czt_manager.h"
#include "mo/algorithms/chmcts/ch_thts_chance_node.h"
#include "mo/algorithms/contextual_zooming/czt_chance_node.h"




namespace thts {
    // forward declare 
    class ChCztDNode;
    class MoThtsContext;

    /**
     *  CHMCTS chance node
     * 
     * This code is quite messy, but don't plan to support it long term, sorry if you're reading this
    */
    class ChCztCNode : public ChThtsCNode {
        friend ChCztDNode;

        protected:
            std::shared_ptr<CztCNode> czt_node;

        public:
            ChCztCNode(
                std::shared_ptr<ChCztManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const ChCztDNode> parent=nullptr); 

            virtual ~ChCztCNode() = default;
            
            virtual void visit(MoThtsContext& ctx);
            virtual std::shared_ptr<const State> sample_observation(MoThtsContext& ctx);
            virtual void backup(
                const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
                const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
                const Eigen::ArrayXd trial_cumulative_return_after_node, 
                const Eigen::ArrayXd trial_cumulative_return,
                MoThtsContext& ctx);

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
            std::shared_ptr<ChCztDNode> create_child_node(std::shared_ptr<const State> next_state);
            virtual std::shared_ptr<ChThtsDNode> create_child_node_helper(
                std::shared_ptr<const State> next_state) const override;
            std::shared_ptr<ChCztDNode> get_child_node(std::shared_ptr<const State> next_state) const;



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