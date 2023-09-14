#include "algorithms/ments/dbments_decision_node.h"

#include "helper_templates.h"

using namespace std; 

namespace thts {
    DBMentsDNode::DBMentsDNode(
        shared_ptr<MentsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const DBMentsCNode> parent) :
            MentsDNode(
                thts_manager,
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MentsCNode>(parent)),
            DPDNode(heuristic_value)
    {
    }

    /**
     * Call both ments and dp visit functions
     */
    void DBMentsDNode::visit(ThtsEnvContext& ctx) {
        MentsDNode::visit(ctx);
        DPDNode::visit_dp(is_leaf());
    }

    /**
     * Calls the dpdnode implementation of recommend action
     */
    shared_ptr<const Action> DBMentsDNode::recommend_action_best_dp_value() const {
        if (children.size() == 0u) {
            int index = thts_manager->get_rand_int(0, actions->size());
            return actions->at(index);
        }

        MentsManager& manager = (MentsManager&) *ThtsDNode::thts_manager;
        return DPDNode::recommend_action_best_dp_value<DBMentsCNode>(
            children, *thts_manager, manager.recommend_visit_threshold, is_opponent());
    }

    /**
     * Returns the best value according to avg returns
    */
    shared_ptr<const Action> DBMentsDNode::recommend_action_best_avg_return() const {
        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        unordered_map<shared_ptr<const Action>, double> action_values;

        for (shared_ptr<const Action> action : *actions) {
            if (!has_child_node(action)) continue;
            DBMentsCNode& child_node = (DBMentsCNode&) *get_child_node(action);
            action_values[action] = opp_coeff * child_node.m_avg_return;
        }

        // If no children, best we can do is select a random action to recommend
        if (action_values.size() == 0u) {
            int index = thts_manager->get_rand_int(0, actions->size());
            return actions->at(index);
        }

        return helper::get_max_key_break_ties_randomly(action_values, *thts_manager);
    }

    /**
     * Implements recommend action to call best dp value
     * ++ hacky recommend with avg return implementation
    */
    shared_ptr<const Action> DBMentsDNode::recommend_action(ThtsEnvContext& ctx) const {
        MentsManager& manager = (MentsManager&) *thts_manager;
        if (manager.recommend_most_visited) {
            return recommend_action_most_visited();
        }
        if (manager.use_avg_return) {
            return recommend_action_best_avg_return();
        }
        return recommend_action_best_dp_value();
    }

    /**
     * Calls both the ments soft backup and dp backup
     * 
     * Recall that the dp backup needs to be passed the type of the child nodes (so can keep dp logic in dp node)
     * 
     * ++ hacky using avg_returns backup impl
     */
    void DBMentsDNode::backup(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsEnvContext& ctx) 
    {
        MentsManager& manager = (MentsManager&) *thts_manager;
        if (!manager.use_avg_return) {
            backup_soft(ctx);
            shared_ptr<const Action> selected_action = nullptr;
            double child_value = 0.0;
            if (manager.use_max_heap) {
                shared_ptr<const Action> selected_action = ctx.get_value_ptr_const<Action>(_action_selected_key);
                DBMentsCNode& child = (DBMentsCNode&) *get_child_node(selected_action);
                child_value = child.dp_value;
            }
            backup_dp<DBMentsCNode>(children, is_opponent(), selected_action, child_value);
            if (manager.alias_use_caching) {
                backup_update_alias_tables(ctx);
            }
            return;
        }

        MentsDNode::backup(
            trial_rewards_before_node, 
            trial_rewards_after_node, 
            trial_cumulative_return_after_node, 
            trial_cumulative_return, ctx);
    }

    /**
     * Return string of the soft value
     */
    string DBMentsDNode::get_pretty_print_val() const {
        stringstream ss;
        ss << dp_value << "(temp:" << get_temp() << ",soft_val:" << soft_value << ")";
        return ss.str();
    }

    /**
     * Make child node
     */
    shared_ptr<DBMentsCNode> DBMentsDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<DBMentsCNode>(
            static_pointer_cast<MentsManager>(ThtsDNode::thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const DBMentsDNode>(shared_from_this()));
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    shared_ptr<ThtsCNode> DBMentsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<DBMentsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}