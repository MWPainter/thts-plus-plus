#pragma once

#include "mo/algorithms/contextual_zooming/bl_thts_decision_node.h"
#include "mo/algorithms/contextual_zooming/czt_manager.h"
#include "mo/algorithms/contextual_zooming/czt_chance_node.h"





namespace thts {
    // forward declare 
    class CztCNode;
    class CztManager;
    class MoThtsContext;

    /**
     * CZT impl
    */
    class CztDNode : public BlThtsDNode {
        friend CztCNode;

        private:
            std::string _action_ctx_key;
            std::string _ball_ctx_key;

        public:
            CztDNode(
                std::shared_ptr<CztManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const CztCNode> parent=nullptr,
                bool eval_mo_heuristic=true); 

            virtual ~CztDNode() = default;
            
            virtual void visit(MoThtsContext& ctx)  override;
            virtual std::shared_ptr<const Action> select_action(MoThtsContext& ctx)  override;
            virtual std::shared_ptr<const Action> recommend_action(MoThtsContext& ctx) const  override;
            virtual void backup(
                const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
                const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
                const Eigen::ArrayXd trial_cumulative_return_after_node, 
                const Eigen::ArrayXd trial_cumulative_return,
                MoThtsContext& ctx)  override;

        protected:
            virtual std::string get_pretty_print_val() const override;

        private:
            // N term for CZT (estimate for total number of trials (that will be run in total))
            // Either the number of visits to this node, or, min_k 2^k s.t. num_visits < 2^k
            double get_N_term(MoThtsContext& ctx) const;

            // Fills "CZ values" used to select action
            void fill_cz_values_and_ball_ptrs(
                ActionVector& actions,
                std::unordered_map<std::shared_ptr<const Action>,double>& cz_values, 
                std::unordered_map<std::shared_ptr<const Action>,std::shared_ptr<CzBall>>& cz_balls, 
                MoThtsContext& ctx);

        public:
            virtual ConvexHull get_convex_hull() const override;
        


        /**
         * Boilerplate function definitions. 
         * 
         * Functionality implemented in thts_decision_node.h, but it's useful to have wrappers to avoid needing to 
         * use pointer casts frequently.
         * 
         * Boilerplate implementations provided in thts_decision_node_template.h
         */
        public:
            std::shared_ptr<CztCNode> create_child_node(std::shared_ptr<const Action> action);
            virtual std::shared_ptr<BlThtsCNode> create_child_node_helper(
                std::shared_ptr<const Action> action) const override;
            std::shared_ptr<CztCNode> get_child_node(std::shared_ptr<const Action> action) const;
    };
}