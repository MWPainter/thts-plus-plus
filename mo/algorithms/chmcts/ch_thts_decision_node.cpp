#include "mo/algorithms/chmcts/ch_thts_decision_node.h"

#include "helper_templates.h"

using namespace std; 

namespace thts {
    ChThtsDNode::ChThtsDNode(
        shared_ptr<ChThtsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ChThtsCNode> parent) :
            MoThtsDNode(
                static_pointer_cast<MoThtsManager>(thts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MoThtsCNode>(parent)),
            num_backups(0),
            convex_hull(mo_heuristic_value)
    {
    }
    
    void ChThtsDNode::visit(MoThtsContext& ctx) 
    {
        MoThtsDNode::visit_itfc(ctx);
        // num_visits += 1;
    } 

    /**
     * Convex hull is initialised with a single point tagged with nullptr
     * Return a random action if the best point tag is nullptr
    */
    shared_ptr<const Action> ChThtsDNode::recommend_action(MoThtsContext& ctx) const 
    {  
        unordered_map<shared_ptr<const Action>,double> utilities;
        fill_contextual_q_values(utilities, ctx, numeric_limits<double>::min());
        return thts::helper::get_max_key_break_ties_randomly(utilities, *thts_manager);
    }
    // shared_ptr<const Action> ChThtsDNode::recommend_action(MoThtsContext& ctx) const 
    // {  
    //     unordered_map<shared_ptr<const Action>,double> utilities;
    //     for (const pair<const shared_ptr<const Action>,shared_ptr<ThtsCNode>>& child_pair : children) {
    //         shared_ptr<const Action> action = child_pair.first;
    //         ChThtsCNode& ch_child = (ChThtsCNode&) *child_pair.second;
    //         lock_guard<mutex> lg(ch_child.get_lock()); 
    //         utilities[action] = ch_child.convex_hull.get_max_linear_utility(ctx.context_weight);
    //     }  
        
    //     // If no children, act randomly
    //     if (utilities.size() == 0) {
    //         shared_ptr<ActionVector> actions = thts_manager->thts_env()->get_valid_actions_itfc(state, ctx);
    //         int index = thts_manager->get_rand_int(0, actions->size());
    //         return actions->at(index);
    //     }

    //     // Return best utility
    //     return thts::helper::get_max_key_break_ties_randomly(utilities, *thts_manager);
    // }
 
    void ChThtsDNode::backup(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        MoThtsContext& ctx)
    {
        increment_and_update_backup_count();

        convex_hull = ConvexHull();
        for (pair<const shared_ptr<const Action>,shared_ptr<ThtsCNode>>& child_pair : children) {
            ChThtsCNode& ch_child = (ChThtsCNode&) *child_pair.second;
            lock_guard<mutex> lg(ch_child.get_lock()); 
            convex_hull |= ch_child.convex_hull;
        }  

        // remember to incr num_backups
        num_backups++;
    }

    double ChThtsDNode::get_contextual_q_value(const MoThtsContext& ctx) {
        return convex_hull.get_max_linear_utility(ctx.context_weight);
    }

    void ChThtsDNode::fill_contextual_q_values(
        unordered_map<shared_ptr<const Action>,double>& q_values, 
        MoThtsContext& ctx, 
        double default_q_value) const
    {
        ThtsEnv& env = *thts_manager->thts_env();
        shared_ptr<ActionVector> actions = env.get_valid_actions_itfc(state,ctx);
        for (shared_ptr<const Action> action : *actions) {
            if (!has_child_node_itfc(action)) {
                q_values[action] = default_q_value;
                continue;
            }
            ChThtsCNode& child = (ChThtsCNode&) *get_child_node_itfc(action);
            lock_guard<mutex> lg(child.node_lock);
            q_values[action] = child.get_contextual_q_value(ctx.context_weight);
        }
    }

    string ChThtsDNode::get_convex_hull_pretty_print_string() const
    {
        stringstream ss;
        ss << convex_hull;
        return ss.str();
    }

    ConvexHull ChThtsDNode::get_convex_hull() const {
        return convex_hull;
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<ChThtsCNode> ChThtsDNode::create_child_node(shared_ptr<const Action> action) 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::create_child_node_itfc(act_itfc);
        return static_pointer_cast<ChThtsCNode>(new_child);
    }

    shared_ptr<ChThtsCNode> ChThtsDNode::get_child_node(shared_ptr<const Action> action) const 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::get_child_node_itfc(act_itfc);
        return static_pointer_cast<ChThtsCNode>(new_child);
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    void ChThtsDNode::visit_itfc(ThtsContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        visit(ctx_itfc);
    }

    shared_ptr<const Action> ChThtsDNode::select_action_itfc(ThtsContext& ctx) 
    {
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        return select_action(mo_ctx);
    }

    shared_ptr<const Action> ChThtsDNode::recommend_action_itfc(ThtsContext& ctx) const 
    {
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        return recommend_action(mo_ctx);
    }

    void ChThtsDNode::backup_itfc(
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

    shared_ptr<ThtsCNode> ChThtsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ChThtsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}