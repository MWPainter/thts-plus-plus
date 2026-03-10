#include "algorithms/ments/rents/rents_decision_node.h"

#include "helper_templates.h"

#include <cmath>
#include <limits>
#include <sstream>

using namespace std; 

static double EPS = 1e-16;

namespace thts {
    RentsDNode::RentsDNode(
        shared_ptr<MentsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const RentsCNode> parent) :
            MentsDNode(
                thts_manager,
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MentsCNode>(parent))
    {
    }

    /**
        Keeping track of the current selection depth.
    */
    int RentsDNode::get_and_increment_current_selection_depth(ThtsContext& ctx) const
    {   
        // root node case
        if (!ctx.context_map_contains(CURRENT_SELECTION_DEPTH_KEY)) 
        {   
            // No value means we are at depth 0, so increment to 1
            ctx.put_value<int>(CURRENT_SELECTION_DEPTH_KEY, make_shared<int>(1));
            return 0;
        }

        // Otherwise, get it out of the map, and replace it with incremented value
        int current_selection_depth = *ctx.get_value_ptr<int>(CURRENT_SELECTION_DEPTH_KEY);
        ctx.put_value<int>(CURRENT_SELECTION_DEPTH_KEY, make_shared<int>(current_selection_depth+1));
        return current_selection_depth;
    }

    /**
        Just a unique id for this node
    */
    string RentsDNode::get_node_id_string() const 
    {
        stringstream ss;
        ss << static_cast<const void*>(this);
        return ss.str();
    }

    /**
        Unique id, tagged with an integer (depth)
    */
    string RentsDNode::get_node_at_depth_id_string(int current_selection_depth) const
    {
        stringstream ss;
        ss << get_node_id_string() << "_" << current_selection_depth;
        return ss.str();
    }

    /**
        Gets the last distribution used in this trial (only will work during selection)
    */
    shared_ptr<ActionDistr> RentsDNode::get_parent_distribution_selection(ThtsContext& ctx) const
    {   
        if (!ctx.context_map_contains(PARENT_ACTION_DISTRIBUTION_KEY)) {
            return nullptr;
        }
        return ctx.get_value_ptr<ActionDistr>(PARENT_ACTION_DISTRIBUTION_KEY);
    }

    /**
        Gets the parent distribution during backup
    */
    shared_ptr<ActionDistr> RentsDNode::get_parent_distribution_backup(ThtsContext& ctx) const
    {
        string node_id_string = get_node_id_string();
        shared_ptr<string> parent_id_at_depth_string_ptr = ctx.get_value_ptr<string>(node_id_string);
        if (parent_id_at_depth_string_ptr == nullptr) 
        {
            return nullptr;
        }
        return ctx.get_value_ptr<ActionDistr>(*parent_id_at_depth_string_ptr);
    }

    shared_ptr<ActionDistr> RentsDNode::get_parent_distribution(bool for_backup, ThtsContext& ctx) const
    {
        if (for_backup) 
        {
            return get_parent_distribution_backup(ctx);
        }
        return get_parent_distribution_selection(ctx);
    }

    /**
        Stores the action distribution for this node in the context
        This is non-trivial, because we need to update all parts of the context mapping
    */
    void RentsDNode::update_context_after_selection(int current_selection_depth, shared_ptr<ActionDistr> action_distr, ThtsContext& ctx) const
    {
        // Get the current parent_id_at_depth_string
        shared_ptr<string> parent_id_at_depth_string_ptr = nullptr;
        if (ctx.context_map_contains(CURRENT_PARENT_ID_AT_DEPTH_STRING_KEY)) {
            parent_id_at_depth_string_ptr = ctx.get_value_ptr<string>(CURRENT_PARENT_ID_AT_DEPTH_STRING_KEY);
        }

        // Get current node ids
        string node_id_string = get_node_id_string();
        string node_at_depth_id_string = get_node_at_depth_id_string(current_selection_depth);

        // Update context for next selection
        ctx.put_value<string>(CURRENT_PARENT_ID_AT_DEPTH_STRING_KEY, make_shared<string>(node_at_depth_id_string));
        ctx.put_value<ActionDistr>(PARENT_ACTION_DISTRIBUTION_KEY, action_distr);

        // Update context for backup at this node
        ctx.put_value<string>(node_id_string, parent_id_at_depth_string_ptr);
        ctx.put_value<ActionDistr>(node_at_depth_id_string, action_distr);
    }

    /**
    * Get prob from parent distribution (handling boundary cases at the root node and when parent didn't have the 
    * action passed in as an option).
    * 
    * Just returns the value stored in the distribution.
    * If the action is not in the distribution return 0.0
    * If the distribution is nullptr (then we are root node) and return 1.0 so can compute normal ments distr.
    */
    double RentsDNode::get_parent_action_prob(
        shared_ptr<ActionDistr> parent_distr, shared_ptr<const Action> action) const 
    {
        if (parent_distr == nullptr) return 1.0;
        if (parent_distr->find(action) == parent_distr->end()) return 0.0;
        return parent_distr->at(action);
    }

    /**
     * Compute action weights.
     * 
     * Computes the distribution from the paper 
     * Paper: http://proceedings.mlr.press/v139/dam21a/dam21a.pdf
     * 
     * Multiplies weights from parent decision node into weights for actions at this node.
     */
    void RentsDNode::compute_action_weights(
        ActionDistr& action_weights, 
        double& sum_action_weights, 
        double& normalisation_term, 
        ThtsContext& context,
        bool for_backup,
        bool only_actions_with_children) const
    {
        // get temp
        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        double temp = get_temp();

        // Get current q values
        unordered_map<shared_ptr<const Action>,double> q_values;
        fill_soft_q_values(q_values, opp_coeff, for_backup);

        // If for backup, remove the q values that didn't come from an updated child node
        if (only_actions_with_children) 
        {
            for (shared_ptr<const Action> action : *actions) 
            {
                if (!has_child_node(action)) continue;
                MentsCNode& child = (MentsCNode&) *get_child_node(action);
                if (child.get_num_backups() > 0) continue;
                q_values.erase(action);
            }
        }

        // optionally normalise q values
        MentsManager& manager = (MentsManager&) *thts_manager;
        if (!for_backup && manager.normalise_q_values) {
            double min_q_value = numeric_limits<double>::max();
            double max_q_value = numeric_limits<double>::lowest();

            for (pair<shared_ptr<const Action>,double> pr : q_values) {
                double q_value = pr.second;
                if (q_value < min_q_value) min_q_value = q_value;
                if (q_value > max_q_value) max_q_value = q_value;
            }

            for (pair<shared_ptr<const Action>,double> pr : q_values) {
                shared_ptr<const Action> action = pr.first;
                double q_value = pr.second;
                q_values[action] = (q_value - min_q_value) / (max_q_value - min_q_value + EPS);
            }
        }

        // compute normalisation term
        normalisation_term = numeric_limits<double>::lowest();
        for (pair<shared_ptr<const Action>,double> pair : q_values) {
            double q_value = pair.second;
            double q_value_over_temp =  q_value / temp;
            if (q_value_over_temp > normalisation_term) {
                normalisation_term = q_value_over_temp;
            }
        }

        // Get parent distribution
        shared_ptr<ActionDistr> parent_distr = get_parent_distribution(for_backup, context);

        // compute action weights
        sum_action_weights = 0.0;
        for (pair<shared_ptr<const Action>,double> pr : q_values) {
            shared_ptr<const Action> action = pr.first;
            double soft_q_value = pr.second;
            if (!std::isfinite(soft_q_value)) continue;
            double action_weight = exp((soft_q_value/temp) - normalisation_term);
            if (!std::isfinite(action_weight)) continue;
            action_weight *= get_parent_action_prob(parent_distr, action);
            action_weights[action] = action_weight;
            sum_action_weights += action_weight;
        }
        
        // If all action weights extremely small, then just make it uniform random, for numerical stability
        if (sum_action_weights < EPS) {
            double uniform_weight = 1.0 / actions->size();
            for (shared_ptr<const Action> action : *actions) {
                action_weights[action] = uniform_weight;
            }
            sum_action_weights = 1.0;
        }
    }

    /**
     * Implements selct action for rents
     * 
     * - Computes the action distribution.
     * - Stores the distribution in the context
     * - Samples an action
     * - Creates the node if it doesn't exist already
     */
    shared_ptr<const Action> RentsDNode::select_action_rents(ThtsContext& ctx) {
        shared_ptr<ActionDistr> action_distr = make_shared<ActionDistr>();
        compute_action_distribution(*action_distr, ctx);

        int current_selection_depth = get_and_increment_current_selection_depth(ctx);
        update_context_after_selection(current_selection_depth, action_distr, ctx);

        shared_ptr<const Action> selected_action = helper::sample_from_distribution(*action_distr, *thts_manager);
        if (!has_child_node(selected_action)) {
            create_child_node(selected_action);
        }
        return selected_action;
    }

    /**
     * Calls the rents implementation of select action
     */
    shared_ptr<const Action> RentsDNode::select_action(ThtsContext& ctx) {
        return select_action_rents(ctx);
    }

    /**
     * Make child node
     */
    shared_ptr<RentsCNode> RentsDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<RentsCNode>(
            static_pointer_cast<MentsManager>(thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const RentsDNode>(shared_from_this()));
    }
}



/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    shared_ptr<ThtsCNode> RentsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<RentsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}