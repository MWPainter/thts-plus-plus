#include "mo/chmcts_decision_node.h"

using namespace std; 

namespace thts {
    ChmctsDNode::ChmctsDNode(
        shared_ptr<ChmctsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ChmctsCNode> parent) :
            CH_MoThtsDNode(
                static_pointer_cast<CH_MoThtsManager>(thts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const CH_MoThtsCNode>(parent)),
            czt_node(
                make_shared<CztDNode>(
                    static_pointer_cast<CztManager>(thts_manager),
                    state,
                    decision_depth,
                    decision_timestep)) // not passing parent pointer because CZT doesnt use it
    {
    }
    
    void ChmctsDNode::visit(MoThtsContext& ctx) 
    {
        CH_MoThtsDNode::visit(ctx);
        czt_node->visit(ctx);
    } 

    shared_ptr<const Action> ChmctsDNode::select_action(MoThtsContext& ctx)
    {
        shared_ptr<const Action> selected_act = czt_node->select_action(ctx);
        if (!has_child_node_itfc(selected_act)) {
            create_child_node(selected_act);
        }
        return selected_act;
    }

    shared_ptr<const Action> ChmctsDNode::recommend_action(MoThtsContext& ctx) const
    {
        return CH_MoThtsDNode::recommend_action(ctx);
    }

    void ChmctsDNode::backup(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        MoThtsContext& ctx)
    {
        CH_MoThtsDNode::backup(
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

    string ChmctsDNode::get_pretty_print_val() const 
    {
        return "";
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<ChmctsCNode> ChmctsDNode::create_child_node(shared_ptr<const Action> action) 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::create_child_node_itfc(act_itfc);
        return static_pointer_cast<ChmctsCNode>(new_child);
    } 

    /**
     * Added making the child's czt_node pointing to the same CztCNode as our czt_node
    */
    shared_ptr<CH_MoThtsCNode> ChmctsDNode::create_child_node_helper(shared_ptr<const Action> action) const 
    {   
        shared_ptr<ChmctsCNode> child_node = make_shared<ChmctsCNode>(
            static_pointer_cast<ChmctsManager>(thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const ChmctsDNode>(shared_from_this()));
        child_node->czt_node = static_pointer_cast<CztCNode>(czt_node->get_child_node_itfc(action));
        return static_pointer_cast<CH_MoThtsCNode>(child_node);
    }

    shared_ptr<ChmctsCNode> ChmctsDNode::get_child_node(shared_ptr<const Action> action) const 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::get_child_node_itfc(act_itfc);
        return static_pointer_cast<ChmctsCNode>(new_child);
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    void ChmctsDNode::visit_itfc(ThtsEnvContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        visit(ctx_itfc);
    }

    shared_ptr<const Action> ChmctsDNode::select_action_itfc(ThtsEnvContext& ctx) 
    {
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        return select_action(mo_ctx);
    }

    shared_ptr<const Action> ChmctsDNode::recommend_action_itfc(ThtsEnvContext& ctx) const 
    {
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        return recommend_action(mo_ctx);
    }

    void ChmctsDNode::backup_itfc(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        ThtsEnvContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        backup(
            trial_rewards_before_node, 
            trial_rewards_after_node, 
            trial_cumulative_return_after_node, 
            trial_cumulative_return, 
            ctx_itfc);
    }

    shared_ptr<ThtsCNode> ChmctsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<CH_MoThtsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}