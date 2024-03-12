#pragma once

#include "mo/chmcts_decision_node.h"

#include "mo/chmcts_manager.h"
#include "mo/ch_thts_chance_node.h"
#include "mo/czt_chance_node.h"




namespace thts {
    // forward declare 
    class ChmctsDNode;
    class MoThtsContext;

    /**
     *  CHMCTS chance node
     * 
     * This code is quite messy, but don't plan to support it long term, sorry if you're reading this
    */
    class ChmctsCNode : public CH_MoThtsCNode {
        friend ChmctsDNode;

        protected:
            std::shared_ptr<CztCNode> czt_node;

        public:
            ChmctsCNode(
                std::shared_ptr<ChmctsManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const ChmctsDNode> parent=nullptr); 

            virtual ~ChmctsCNode() = default;
            
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
            std::shared_ptr<ChmctsDNode> create_child_node(std::shared_ptr<const State> next_state);
            virtual std::shared_ptr<CH_MoThtsDNode> create_child_node_helper(
                std::shared_ptr<const State> next_state) const override;
            std::shared_ptr<ChmctsDNode> get_child_node(std::shared_ptr<const State> next_state) const;



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