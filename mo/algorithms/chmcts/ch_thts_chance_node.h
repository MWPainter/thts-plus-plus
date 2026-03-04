#pragma once

#include "mo/algorithms/chmcts/ch_thts_decision_node.h"

#include "mo/data_structures/convex_hull.h"
#include "mo/algorithms/chmcts/ch_thts_manager.h"
#include "mo/mo_thts_chance_node.h"

#include <Eigen/Dense>




namespace thts {
    // forward declare 
    class ChThtsDNode;
    class MoThtsContext;

    /**
     * Base class for decision nodes that use ConvexHull objects for their state
    */
    class ChThtsCNode : public MoThtsCNode {
        friend ChThtsDNode;

        protected:
            int num_backups;
            ConvexHull convex_hull;
            ConvexHull convex_hull_for_search;
            Eigen::ArrayXd local_reward;

        public:
            ChThtsCNode(
                std::shared_ptr<ChThtsManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const ChThtsDNode> parent=nullptr); 

            virtual ~ChThtsCNode() = default;
            
            virtual void visit(MoThtsContext& ctx);
            virtual std::shared_ptr<const State> sample_observation(MoThtsContext& ctx) = 0;
            virtual void backup(
                const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
                const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
                const Eigen::ArrayXd trial_cumulative_return_after_node, 
                const Eigen::ArrayXd trial_cumulative_return,
                MoThtsContext& ctx);

            std::string get_convex_hull_pretty_print_string() const;

        protected:
            virtual std::shared_ptr<ChThtsDNode> create_child_node_helper(
                std::shared_ptr<const State> state) const = 0;
            virtual std::string get_pretty_print_val() const override = 0;
        

            double get_contextual_q_value(const MoThtsContext& ctx, bool for_search=true) const;


        /**
         * Boilerplate function definitions. 
         * 
         * Functionality implemented in thts_decision_node.h, but it's useful to have wrappers to avoid needing to 
         * use pointer casts frequently.
         * 
         * Boilerplate implementations provided in thts_decision_node_template.h
         */
        public:
            std::shared_ptr<ChThtsDNode> create_child_node(std::shared_ptr<const State> next_state);
            std::shared_ptr<ChThtsDNode> get_child_node(std::shared_ptr<const State> next_state) const;



        /**
         * ThtsDNode interface function definitions, used by thts subroutines to interact with this node. Copied from 
         * thts_decision_node.h. 
         * 
         * Boilerplate definitions are provided in thts_decision_node_template.h, that wrap above functions in pointer 
         * casts.
         */
        public:
            virtual void visit_itfc(ThtsContext& ctx) override;
            virtual std::shared_ptr<const Observation> sample_observation_itfc(ThtsContext& ctx) override;
            virtual void backup_itfc(
                const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
                const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
                const Eigen::ArrayXd trial_cumulative_return_after_node, 
                const Eigen::ArrayXd trial_cumulative_return,
                ThtsContext& ctx) override;

            virtual std::shared_ptr<ThtsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation, 
                std::shared_ptr<const State> next_state=nullptr) const override;
    };
}