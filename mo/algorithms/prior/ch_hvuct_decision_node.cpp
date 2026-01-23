#include "mo/algorithms/prior/ch_hvuct_decision_node.h"

#include "helper_templates.h"

using namespace std; 

namespace thts {
    ChHvUctDNode::ChHvUctDNode(
        shared_ptr<ChHvUctManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ChHvUctCNode> parent) :
            ChUctDNode(
                static_pointer_cast<ChUctManager>(thts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const ChUctCNode>(parent))
    {
    }

    void ChHvUctDNode::fill_ucb_q_values(ActionDistr& ucb_q_values, MoThtsContext& ctx) const
    {
        ChHvUctManager& manager = (ChHvUctManager&) *thts_manager;
        for (pair<shared_ptr<const Action>,shared_ptr<ThtsCNode>> pair : children) {
            shared_ptr<const Action> action = pair.first;
            ChHvUctCNode& child = (ChHvUctCNode&) *get_child_node(action);
            ucb_q_values[action] = child.convex_hull.hypervolume(*manager.hv_reference_point) / child.num_visits;
        }
    }

    string ChHvUctDNode::get_pretty_print_val() const 
    {
        return "";
    }

    void ChHvUctDNode::update_solved_labelling_confidence_interval_range()
    {
        throw runtime_error("Not implemented yet");
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
    shared_ptr<ChThtsCNode> ChHvUctDNode::create_child_node_helper(shared_ptr<const Action> action) const 
    {   
        shared_ptr<ChHvUctCNode> child_node = make_shared<ChHvUctCNode>(
            static_pointer_cast<ChHvUctManager>(thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const ChHvUctDNode>(shared_from_this()));
        return static_pointer_cast<ChThtsCNode>(child_node);
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    void ChHvUctDNode::visit_itfc(ThtsContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        visit(ctx_itfc);
    }

    shared_ptr<const Action> ChHvUctDNode::select_action_itfc(ThtsContext& ctx) 
    {
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        return select_action(mo_ctx);
    }

    shared_ptr<const Action> ChHvUctDNode::recommend_action_itfc(ThtsContext& ctx) const 
    {
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        return recommend_action(mo_ctx);
    }

    void ChHvUctDNode::backup_itfc(
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

    shared_ptr<ThtsCNode> ChHvUctDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ChThtsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}