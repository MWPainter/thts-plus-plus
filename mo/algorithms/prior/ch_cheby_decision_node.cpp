#include "mo/algorithms/prior/ch_cheby_decision_node.h"

#include "helper_templates.h"

#include <algorithm>
#include <cmath>

using namespace std; 

namespace thts {
    ChChebyUctDNode::ChChebyUctDNode(
        shared_ptr<ChChebyUctManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ChChebyUctCNode> parent) :
            ChUctDNode(
                static_pointer_cast<ChUctManager>(thts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const ChUctCNode>(parent))
    {
    }

    double sigmoid(double x) 
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    /**
     * Gets the Chebychev value for a given convex hull and context
     * 
     * Standard scalarization = use standard cheby scalarization, with a globally set (lower bound/pareto dominated) reference point and convex hull shifted by running reward to be correct relevant to global reference point
     * Prior scalarization = use scalarization from prior work, using a reference point that is an upper bound/pareto dominates all points in the convex hull
     *
     * When using prior scalarization, they do something weird with negating and then passing through a sigmoid function.
     */
    double ChChebyUctDNode::get_cheby_value(const ConvexHull& convex_hull, const MoThtsContext& ctx) const
    {
        ChChebyUctManager& manager = (ChChebyUctManager&) *thts_manager;

        if (manager.use_standard_cheby_scalarization) {
            Vec running_reward = Vec::Zero(manager.reward_dim);
            if (ctx.context_map_contains(RUNNING_REWARD_CTX_KEY)) {
                running_reward = ctx.get_value<Vec>(RUNNING_REWARD_CTX_KEY);
            }
            ConvexHull shifted_convex_hull = convex_hull.add(running_reward);
            return this->cheby_scalarization_value(shifted_convex_hull, *manager.standard_cheby_reference_point, ctx);
        } else {
            Vec reference_point = this->get_prior_cheby_reference_point(convex_hull);
            return sigmoid(-1.0 * this->cheby_scalarization_value(convex_hull, reference_point, ctx));
        }
    }

    Vec ChChebyUctDNode::get_prior_cheby_reference_point(const ConvexHull& convex_hull) const
    {
        ChChebyUctManager& manager = (ChChebyUctManager&) *thts_manager;
        Eigen::ArrayXd max_point = Eigen::ArrayXd::Zero(manager.reward_dim);
        for (const Vec& point : convex_hull.ch_points) {
            for (int i=0; i<manager.reward_dim; i++) {
                max_point[i] = std::max(max_point[i], point[i]);
            }
        }
        max_point += manager.cheby_delta;
        return Vec(max_point);
    }

    double ChChebyUctDNode::cheby_scalarization_value(const ConvexHull& convex_hull, const Vec& reference_point, const MoThtsContext& ctx) const
    {
        ChChebyUctManager& manager = (ChChebyUctManager&) *thts_manager;
        vector<double> diffs = vector<double>(manager.reward_dim, 0.0);
        for (const Vec& point : convex_hull.ch_points) {
            for (int i=0; i<manager.reward_dim; i++) {
                diffs[i] = ctx.context_weight[i] * std::abs(point[i] - reference_point[i]);
            }
        }
        return *std::max_element(diffs.begin(), diffs.end());
    }

    void ChChebyUctDNode::fill_ucb_q_values(ActionDistr& ucb_q_values, MoThtsContext& ctx) const
    {
        ChChebyUctManager& manager = (ChChebyUctManager&) *thts_manager;
        for (pair<shared_ptr<const Action>,shared_ptr<ThtsCNode>> pair : children) {
            shared_ptr<const Action> action = pair.first;
            ChChebyUctCNode& child = (ChChebyUctCNode&) *get_child_node(action);
            ucb_q_values[action] = this->get_cheby_value(child.convex_hull_local, ctx);
        }
    }

    string ChChebyUctDNode::get_pretty_print_val() const 
    {
        return "";
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    /**
     * Added making the child's czt_node pointing to the same CztCNode as our czt_node
    */
    shared_ptr<ChThtsCNode> ChChebyUctDNode::create_child_node_helper(shared_ptr<const Action> action) const 
    {   
        shared_ptr<ChChebyUctCNode> child_node = make_shared<ChChebyUctCNode>(
            static_pointer_cast<ChChebyUctManager>(thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const ChChebyUctDNode>(shared_from_this()));
        return static_pointer_cast<ChThtsCNode>(child_node);
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    void ChChebyUctDNode::visit_itfc(ThtsContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        visit(ctx_itfc);
    }

    shared_ptr<const Action> ChChebyUctDNode::select_action_itfc(ThtsContext& ctx) 
    {
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        return select_action(mo_ctx);
    }

    shared_ptr<const Action> ChChebyUctDNode::recommend_action_itfc(ThtsContext& ctx) const 
    {
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        return recommend_action(mo_ctx);
    }

    void ChChebyUctDNode::backup_itfc(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        ThtsContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        backup(
            trial_rewards_before_node, 
            trial_rewards_after_node, 
            trial_cumulative_return_after_node, 
            trial_cumulative_return, 
            ctx_itfc);
    }

    shared_ptr<ThtsCNode> ChChebyUctDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ChThtsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}