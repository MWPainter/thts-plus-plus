#include "mo/ch_thts_decision_node.h"

using namespace std; 

namespace thts {
    CH_MoThtsDNode::CH_MoThtsDNode(
        shared_ptr<CH_MoThtsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const CH_MoThtsCNode> parent) :
            MoThtsDNode(
                static_pointer_cast<MoThtsManager>(thts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MoThtsCNode>(parent)),
            num_backups(0),
            convex_hull(mo_heuristic_value, nullptr)
    {
    }
    
    void CH_MoThtsDNode::visit(MoThtsContext& ctx) 
    {
        num_visits += 1;
    } 

    /**
     * Convex hull is initialised with a single point tagged with nullptr
     * Return a random action if the best point tag is nullptr
    */
    shared_ptr<const Action> CH_MoThtsDNode::recommend_action(MoThtsContext& ctx) const 
    {  
        unordered_map<shared_ptr<const Action>,double> utilities;
        for (const pair<const shared_ptr<const Action>,shared_ptr<ThtsCNode>>& child_pair : children) {
            shared_ptr<const Action> action = child_pair.first;
            CH_MoThtsCNode& ch_child = (CH_MoThtsCNode&) *child_pair.second;
            lock_guard<mutex> lg(ch_child.get_lock()); 
            utilities[action] = ch_child.convex_hull.get_max_linear_utility(ctx.context_weight);
        }  
        
        // If no children, act randomly
        if (utilities.size() == 0) {
            shared_ptr<ActionVector> actions = thts_manager->thts_env()->get_valid_actions_itfc(state, ctx);
            int index = thts_manager->get_rand_int(0, actions->size());
            return actions->at(index);
        }

        // Return best utility
        return thts::helper::get_max_key_break_ties_randomly(utilities, *thts_manager);
    }

    /**
     * N.B. This is how we used to do it, but was the wrong idea
     * 
     * Convex hull is initialised with a single point tagged with nullptr
     * Return a random action if the best point tag is nullptr
    */
    shared_ptr<const Action> CH_MoThtsDNode::recommend_action_from_tags(MoThtsContext& ctx) const 
    {  
        shared_ptr<const Action> act = convex_hull.get_best_point_tag(ctx.context_weight, *thts_manager);
        if (act == nullptr) {
            shared_ptr<ActionVector> actions = thts_manager->thts_env()->get_valid_actions_itfc(state, ctx);
            int index = thts_manager->get_rand_int(0, actions->size());
            act = actions->at(index);
        }
        return act;
    }
 
    void CH_MoThtsDNode::backup(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        MoThtsContext& ctx)
    {
        convex_hull = ConvexHull<shared_ptr<const Action>>();
        for (pair<const shared_ptr<const Action>,shared_ptr<ThtsCNode>>& child_pair : children) {
            CH_MoThtsCNode& ch_child = (CH_MoThtsCNode&) *child_pair.second;
            lock_guard<mutex> lg(ch_child.get_lock()); 
            convex_hull |= ch_child.convex_hull;
        }  

        // remember to incr num_backups
        num_backups++;
    }

    string CH_MoThtsDNode::get_convex_hull_pretty_print_string() const
    {
        stringstream ss;
        ss << convex_hull;
        return ss.str();
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<CH_MoThtsCNode> CH_MoThtsDNode::create_child_node(shared_ptr<const Action> action) 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::create_child_node_itfc(act_itfc);
        return static_pointer_cast<CH_MoThtsCNode>(new_child);
    }

    shared_ptr<CH_MoThtsCNode> CH_MoThtsDNode::get_child_node(shared_ptr<const Action> action) const 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::get_child_node_itfc(act_itfc);
        return static_pointer_cast<CH_MoThtsCNode>(new_child);
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    void CH_MoThtsDNode::visit_itfc(ThtsEnvContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        visit(ctx_itfc);
    }

    shared_ptr<const Action> CH_MoThtsDNode::select_action_itfc(ThtsEnvContext& ctx) 
    {
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        return select_action(mo_ctx);
    }

    shared_ptr<const Action> CH_MoThtsDNode::recommend_action_itfc(ThtsEnvContext& ctx) const 
    {
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        return recommend_action(mo_ctx);
    }

    void CH_MoThtsDNode::backup_itfc(
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

    shared_ptr<ThtsCNode> CH_MoThtsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<CH_MoThtsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}