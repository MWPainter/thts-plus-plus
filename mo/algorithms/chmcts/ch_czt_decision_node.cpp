#include "mo/algorithms/chmcts/ch_czt_decision_node.h"

using namespace std; 

namespace thts {
    ChCztDNode::ChCztDNode(
        shared_ptr<ChCztManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ChCztCNode> parent) :
            ChThtsDNode(
                static_pointer_cast<ChThtsManager>(thts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const ChThtsCNode>(parent)),
            czt_node(
                make_shared<CztDNode>(
                    static_pointer_cast<CztManager>(thts_manager),
                    state,
                    decision_depth,
                    decision_timestep,
                    nullptr, // not passing parent pointer because CZT doesnt use it
                    false)) // dont evaluate MO heuristic twice, will cause bug running twice with gym envs
    {
    }
    
    void ChCztDNode::visit(MoThtsContext& ctx) 
    {
        ChThtsDNode::visit(ctx);
        czt_node->visit(ctx);
    } 

    shared_ptr<const Action> ChCztDNode::select_action(MoThtsContext& ctx)
    {
        shared_ptr<const Action> selected_act = czt_node->select_action(ctx);
        if (!has_child_node_itfc(selected_act)) {
            create_child_node(selected_act);
        }
        return selected_act;
    }

    shared_ptr<const Action> ChCztDNode::recommend_action(MoThtsContext& ctx) const
    {
        return ChThtsDNode::recommend_action(ctx);
    }

    void ChCztDNode::backup(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        MoThtsContext& ctx)
    {
        ChThtsDNode::backup(
            trial_rewards_before_node, 
            trial_rewards_after_node, 
            trial_cumulative_return_after_node, 
            trial_cumulative_return,
            ctx);
        czt_node->backup(
            trial_rewards_before_node, 
            trial_rewards_after_node, 
            trial_cumulative_return_after_node, 
            trial_cumulative_return,
            ctx);
    }

    string ChCztDNode::get_pretty_print_val() const 
    {
        return "";
    }

    void ChCztDNode::update_solved_labelling_confidence_interval_range()
    {
        throw runtime_error("Not implemented yet");
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<ChCztCNode> ChCztDNode::create_child_node(shared_ptr<const Action> action) 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::create_child_node_itfc(act_itfc);
        return static_pointer_cast<ChCztCNode>(new_child);
    } 

    /**
     * Added making the child's czt_node pointing to the same CztCNode as our czt_node
    */
    shared_ptr<ChThtsCNode> ChCztDNode::create_child_node_helper(shared_ptr<const Action> action) const 
    {   
        shared_ptr<ChCztCNode> child_node = make_shared<ChCztCNode>(
            static_pointer_cast<ChCztManager>(thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const ChCztDNode>(shared_from_this()));
        child_node->czt_node = static_pointer_cast<CztCNode>(czt_node->get_child_node_itfc(action));
        return static_pointer_cast<ChThtsCNode>(child_node);
    }

    shared_ptr<ChCztCNode> ChCztDNode::get_child_node(shared_ptr<const Action> action) const 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::get_child_node_itfc(act_itfc);
        return static_pointer_cast<ChCztCNode>(new_child);
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    void ChCztDNode::visit_itfc(ThtsContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        visit(ctx_itfc);
    }

    shared_ptr<const Action> ChCztDNode::select_action_itfc(ThtsContext& ctx) 
    {
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        return select_action(mo_ctx);
    }

    shared_ptr<const Action> ChCztDNode::recommend_action_itfc(ThtsContext& ctx) const 
    {
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        return recommend_action(mo_ctx);
    }

    void ChCztDNode::backup_itfc(
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

    shared_ptr<ThtsCNode> ChCztDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ChThtsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}