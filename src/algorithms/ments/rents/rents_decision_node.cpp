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
                static_pointer_cast<const MentsCNode>(parent)),
            _node_distr_key(),
            _parent_distr_key(),
            cached_action_distr()
    {
        stringstream ss_n;
        ss_n << "d_" << decision_depth;
        _node_distr_key = ss_n.str();

        if (decision_depth > 0) {
            stringstream ss_p;
            ss_p << "d_" << decision_depth-1;
            _parent_distr_key = ss_p.str();
        }

        ThtsEnvContext spoof_ctx;
        cached_action_distr = select_action_alias_tables_get_mixed_distr(spoof_ctx)->get_distr_map();
    }

    /**
     * Gets the action distribution for a parent node 
     * Or just null pointer if we're the root node
    */
    shared_ptr<ActionDistr> RentsDNode::get_parent_distr_from_context(ThtsEnvContext& ctx) const {
        if (decision_depth < 1 || !ctx.contains_key(_parent_distr_key)) return nullptr;
        return ctx.get_value_ptr<ActionDistr>(_parent_distr_key);
    }

    /**
     * Puts the action distribution for this node into the thts env context
    */
    void RentsDNode::put_node_distr_in_context(shared_ptr<ActionDistr> action_distr, ThtsEnvContext& ctx) const {
        ctx.put_value(_node_distr_key, action_distr);
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
        ThtsEnvContext& context) const
    {
        // get temp
        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        double temp = get_temp();

        // compute normalisation term
        normalisation_term = numeric_limits<double>::lowest();
        for (shared_ptr<const Action> action : *actions) {
            double q_value_over_temp = get_soft_q_value(action,opp_coeff) / temp;
            if (normalisation_term < q_value_over_temp) {
                normalisation_term = q_value_over_temp;
            }
        }

        // Get parent distribution
        shared_ptr<ActionDistr> parent_distr = get_parent_distr_from_context(context);

        // compute action weights
        sum_action_weights = 0.0;
        for (shared_ptr<const Action> action : *actions) {
            double soft_q_value = get_soft_q_value(action,opp_coeff);
            double action_weight = exp((soft_q_value/temp) - normalisation_term);
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
    shared_ptr<const Action> RentsDNode::select_action_rents(ThtsEnvContext& ctx) {
        shared_ptr<ActionDistr> action_distr = make_shared<ActionDistr>();
        compute_action_distribution(*action_distr, ctx);
        put_node_distr_in_context(action_distr, ctx);
        shared_ptr<const Action> selected_action;
        while (is_nullptr_or_should_skip_under_construction_child(selected_action)) {
            selected_action = helper::sample_from_distribution(*action_distr, *thts_manager, false);
            if (!has_child_node(selected_action)) {
                create_child_node(selected_action);
            }
        }
        return selected_action;
    }

    /**
     * Add context stuff to select action alias tables
    */
    shared_ptr<const Action> RentsDNode::select_action_alias_tables(ThtsEnvContext& ctx) {
        // Get mixed distribution
        shared_ptr<MixedDistribution<shared_ptr<const Action>>> mixed_distr = 
            select_action_alias_tables_get_mixed_distr(ctx);

        // Put cached distribution in context
        put_node_distr_in_context(cached_action_distr, ctx);

        // Sample, and handle making child if need be, return
        MentsManager& manager = (MentsManager&) *thts_manager;
        shared_ptr<const Action> selected_action;
        while (is_nullptr_or_should_skip_under_construction_child(selected_action)) {
            selected_action = mixed_distr->sample(manager);
            if (!has_child_node(selected_action)) {
                create_child_node(selected_action);
            }
        }
        return selected_action;
    }

    /**
     * Calls the rents implementation of select action
     */
    shared_ptr<const Action> RentsDNode::select_action(ThtsEnvContext& ctx) {
        MentsManager& manager = (MentsManager&) *thts_manager;
        shared_ptr<const Action> selected_action;
        if (manager.alias_use_caching) {
            selected_action = select_action_alias_tables(ctx);
        } else {
            selected_action = select_action_rents(ctx);
        }
        if (manager.use_max_heap) {
            ctx.put_value_const(_action_selected_key, selected_action);
        }
        return selected_action;
    }

    /**
     * Implement soft backup with auxilary variables to make it quick
     * - see MentsDNode.cpp implementation for description of what this is doing
     * - only changes are adding the 'parent_prob' into the 'child_term'
    */
    void RentsDNode::backup_soft_with_max_heap(ThtsEnvContext& ctx) {
        num_backups++;

        shared_ptr<const Action> selected_action = ctx.get_value_ptr_const<Action>(_action_selected_key);
        RentsCNode& child = (RentsCNode&) *get_child_node(selected_action);
        lock_guard<mutex> lg(child.node_lock);
        
        double old_child_term = sum_exp_child_terms[selected_action];
        double old_max_value = 0.0;
        if (max_heap->size() > 0) old_max_value = max_heap->peek_top_value();

        double temp = get_temp();
        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        double child_value = opp_coeff * child.soft_value;
        max_heap->insert_or_assign(selected_action, child_value);
        double max_value = max_heap->peek_top_value();
        double parent_prob = get_parent_action_prob(get_parent_distr_from_context(ctx), selected_action);
        double child_term = parent_prob * exp((child_value - max_value) / temp);
        sum_exp_child_terms[selected_action] = child_term;

        sum_exp_child_values -= old_child_term;
        sum_exp_child_values *= exp((-max_value + old_max_value) / temp);
        sum_exp_child_values += child_term;

        soft_value = opp_coeff * (log(sum_exp_child_values) + max_value / temp);
    }

    /**
     * Update alias tables in backup
     * Update the cached action distribution in rents
    */
    void RentsDNode::backup_update_alias_tables(ThtsEnvContext& ctx) {
        MentsDNode::backup_update_alias_tables(ctx);
        cached_action_distr = select_action_alias_tables_get_mixed_distr(ctx)->get_distr_map();
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
