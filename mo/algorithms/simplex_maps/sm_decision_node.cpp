#include "mo/algorithms/simplex_maps/sm_decision_node.h"

using namespace std; 

namespace thts {
    /**
     * TODO: would like to set initial q values using heuristic value in simplex map, rather than 'default_q_value'
     *      - make the default heuristic function return the default q value though (and default q value is zero)
     *      - should make prior_policy and heuristic_fn objects, that both implement 'operator()' rather than using pointer types
     *      - then we can have heuristic fn's with state as we want
     *      - this needs more editing through the entire library though
     *      - but will be nice to seperate those two things out into seperate files at least
    */
    SmThtsDNode::SmThtsDNode(
        shared_ptr<SmThtsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const SmThtsCNode> parent) :
            MoThtsDNode(
                static_pointer_cast<MoThtsManager>(thts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MoThtsCNode>(parent)),
            // simplex_map(thts_manager->reward_dim, thts_manager->default_q_value)
            simplex_map(thts_manager->reward_dim, mo_heuristic_value.vec)
    {
    }
    
    void SmThtsDNode::visit(MoThtsContext& ctx) 
    {
        MoThtsDNode::visit_itfc(ctx);
        // num_visits += 1;
    } 

    string SmThtsDNode::get_simplex_map_pretty_print_string() const {
        return simplex_map.get_pretty_print_string();
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<SmThtsCNode> SmThtsDNode::create_child_node(shared_ptr<const Action> action) 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::create_child_node_itfc(act_itfc);
        return static_pointer_cast<SmThtsCNode>(new_child);
    }

    shared_ptr<SmThtsCNode> SmThtsDNode::get_child_node(shared_ptr<const Action> action) const 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::get_child_node_itfc(act_itfc);
        return static_pointer_cast<SmThtsCNode>(new_child);
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    void SmThtsDNode::visit_itfc(ThtsContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        visit(ctx_itfc);
    }

    shared_ptr<const Action> SmThtsDNode::select_action_itfc(ThtsContext& ctx) 
    {
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        return select_action(mo_ctx);
    }

    shared_ptr<const Action> SmThtsDNode::recommend_action_itfc(ThtsContext& ctx) const 
    {
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        return recommend_action(mo_ctx);
    }

    void SmThtsDNode::backup_itfc(
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

    shared_ptr<ThtsCNode> SmThtsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<SmThtsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}