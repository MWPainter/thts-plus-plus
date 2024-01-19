#include "mo/bl_thts_decision_node.h"

using namespace std; 

namespace thts {
    BL_MoThtsDNode::BL_MoThtsDNode(
        shared_ptr<BL_MoThtsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const BL_MoThtsCNode> parent) :
            MoThtsDNode(
                static_pointer_cast<MoThtsManager>(thts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MoThtsCNode>(parent)),
            ball_list(thts_manager->reward_dim, thts_manager->num_backups_before_allowed_to_split)
    {
    }
    
    void BL_MoThtsDNode::visit(MoThtsContext& ctx) 
    {
        num_visits += 1;
    } 

    string BL_MoThtsDNode::get_ball_list_pretty_print_string() const {
        return ball_list.get_pretty_print_string();
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<BL_MoThtsCNode> BL_MoThtsDNode::create_child_node(shared_ptr<const Action> action) 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::create_child_node_itfc(act_itfc);
        return static_pointer_cast<BL_MoThtsCNode>(new_child);
    }

    shared_ptr<BL_MoThtsCNode> BL_MoThtsDNode::get_child_node(shared_ptr<const Action> action) const 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::get_child_node_itfc(act_itfc);
        return static_pointer_cast<BL_MoThtsCNode>(new_child);
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    void BL_MoThtsDNode::visit_itfc(ThtsEnvContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        visit(ctx_itfc);
    }

    shared_ptr<const Action> BL_MoThtsDNode::select_action_itfc(ThtsEnvContext& ctx) 
    {
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        return select_action(mo_ctx);
    }

    shared_ptr<const Action> BL_MoThtsDNode::recommend_action_itfc(ThtsEnvContext& ctx) const 
    {
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        return recommend_action(mo_ctx);
    }

    void BL_MoThtsDNode::backup_itfc(
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

    shared_ptr<ThtsCNode> BL_MoThtsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<BL_MoThtsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}