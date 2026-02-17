#pragma once

#include "mo/algorithms/prior/ch_cheby_chance_node.h"

#include "mo/algorithms/prior/ch_cheby_manager.h"
#include "mo/algorithms/chmcts/ch_uct_decision_node.h"




namespace thts {
    // forward declare 
    class ChChebyUctCNode;
    class ChChebyUctManager;
    class MoThtsContext;

    static constexpr std::string RUNNING_REWARD_CTX_KEY = "running_reward";

    /**
     * CHMCTS Decision node
     * 
     * This code is quite messy, but don't plan to support it long term, sorry if you're reading this
    */
    class ChChebyUctDNode : public ChUctDNode {
        friend ChChebyUctCNode;

        protected:

        public:
            ChChebyUctDNode(
                std::shared_ptr<ChChebyUctManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const ChChebyUctCNode> parent=nullptr); 

            virtual ~ChChebyUctDNode() = default;
            
            double get_cheby_value(const ConvexHull& convex_hull, const MoThtsContext& ctx) const;
            Vec get_prior_cheby_reference_point(const ConvexHull& convex_hull) const;
            double cheby_scalarization_value(const ConvexHull& convex_hull, const Vec& reference_point, const MoThtsContext& ctx) const;
            
            // virtual void visit(MoThtsContext& ctx) override;
            // virtual std::shared_ptr<const Action> select_action(MoThtsContext& ctx) override;
            // virtual std::shared_ptr<const Action> recommend_action(MoThtsContext& ctx) const override;
            // virtual void backup(
            //     const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
            //     const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
            //     const Eigen::ArrayXd trial_cumulative_return_after_node, 
            //     const Eigen::ArrayXd trial_cumulative_return,
            //     MoThtsContext& ctx) override;

        protected:
            virtual std::string get_pretty_print_val() const override;

            // double compute_ucb_confidence_interval(int num_visits, int child_visits) const;
            virtual void fill_ucb_q_values(ActionDistr& ucb_q_values, MoThtsContext& ctx) const override;
            // void fill_ucb_values(ActionDistr& ucb_values, MoThtsContext& ctx) const;

        /**
         * Boilerplate function definitions. 
         * 
         * Functionality implemented in thts_decision_node.h, but it's useful to have wrappers to avoid needing to 
         * use pointer casts frequently.
         * 
         * Boilerplate implementations provided in thts_decision_node_template.h
         */
        public:
            // std::shared_ptr<ChChebyUctCNode> create_child_node(std::shared_ptr<const Action> action);
            virtual std::shared_ptr<ChThtsCNode> create_child_node_helper(
                std::shared_ptr<const Action> action) const override;
            // std::shared_ptr<ChChebyUctCNode> get_child_node(std::shared_ptr<const Action> action) const;



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