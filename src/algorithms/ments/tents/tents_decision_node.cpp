#include "algorithms/ments/tents/tents_decision_node.h"

#include "helper_templates.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

using namespace std; 

static double EPS = 1e-16;

namespace thts {
    /** 
     * Constructor, 
     * initialises the maps used by tents,
     * cache the selected_action_key used in contexts
    */
    TentsDNode::TentsDNode(
        shared_ptr<MentsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const TentsCNode> parent) :
            MentsDNode(
                static_pointer_cast<MentsManager>(thts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MentsCNode>(parent))
    {
        for (shared_ptr<const Action> action : *actions) 
        {
            bool for_backup = true;
            double qval = get_soft_q_value_over_temp_safe(action, for_backup);
            qval_to_act.insert(make_pair(qval, action));
            act_to_qval.insert_or_assign(action, qval);
            for_backup = false;
            double qval_for_search = get_soft_q_value_over_temp_safe(action, for_backup);
            qval_to_act_for_search.insert(make_pair(qval_for_search, action));
            act_to_qval_for_search.insert_or_assign(action, qval_for_search);
        }

        stringstream ss;
        ss << decision_depth;
        _selected_action_key = ss.str();
    }

    double TentsDNode::get_temp_safe() const
    {
        double temp = get_temp();
        if (temp <= 0.0 || !std::isfinite(temp)) {
            temp = EPS;
        }
        return temp;
    }

    /**
     * Get the value of Q(s,a)/temp from the best available source (see ments get_soft_q_value, tries child, then prior)
    */
    double TentsDNode::get_soft_q_value_safe(shared_ptr<const Action> action, bool for_backup) const 
    {
        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        double qval = get_soft_q_value(action, opp_coeff, for_backup);
        double temp = get_temp_safe();
        return qval;
    }

    double TentsDNode::get_soft_q_value_over_temp_safe(shared_ptr<const Action> action, bool for_backup) const 
    {
        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        double qval = get_soft_q_value(action, opp_coeff, for_backup);
        double temp = get_temp_safe();
        return qval / temp;
    }

    /**
     * Updates the tents mapping for 'action' to/from 'neq_q_value'
    */
    void TentsDNode::update_maps(shared_ptr<const Action> action, double new_q_value, bool for_backup) 
    {
        if (!for_backup) {
            update_maps_for_search(action, new_q_value);
            return;
        }

        act_to_qval.erase(action);
        for (auto it = qval_to_act.begin(); it != qval_to_act.end(); ++it) {
            if (it->second == action) {
                qval_to_act.erase(it);
                break;
            }
        }

        act_to_qval.insert_or_assign(action, new_q_value);
        qval_to_act.insert(make_pair(new_q_value, action));
    }

    void TentsDNode::update_maps_for_search(shared_ptr<const Action> action, double new_q_value) {
        act_to_qval_for_search.erase(action);
        for (auto it = qval_to_act_for_search.begin(); it != qval_to_act_for_search.end(); ++it) {
            if (it->second == action) {
                qval_to_act_for_search.erase(it);
                break;
            }
        }

        act_to_qval_for_search.insert_or_assign(action, new_q_value);
        qval_to_act_for_search.insert(make_pair(new_q_value, action));
    }

    /**
     * Computes the sparse action set 
     * http://proceedings.mlr.press/v139/dam21a/dam21a.pdf
     * 
     * Basically its the set of actions who's value of Q(s,a)/temp meet the condition in the if statement
     * It is important the the values of Q(s,a)/temp are iterated over from the highest to lowest values (which the 
     * reverse iterator over the multimap will do)
    */
    unique_ptr<ActionVector> TentsDNode::get_sparse_action_set(bool for_backup, bool only_actions_with_children) const {
        if (!for_backup) {
            return get_sparse_action_set_for_search(only_actions_with_children);
        }

        unique_ptr<ActionVector> sparse_action_set = make_unique<ActionVector>();
        double i = 0;
        double sum_values = 0.0;
        for (auto it=qval_to_act.rbegin(); it != qval_to_act.rend(); it++) {
            double value = it->first;
            if (!std::isfinite(value)) continue;
            shared_ptr<const Action> action = it->second;
            if (only_actions_with_children && !has_child_node(action)) continue;
            sum_values += value;
            if (1.0 + (i+1.0)*value > sum_values) {
                sparse_action_set->push_back(action);
            }
            i++;
        }
        return sparse_action_set;
    }

    unique_ptr<ActionVector> TentsDNode::get_sparse_action_set_for_search(bool only_actions_with_children) const {
        unique_ptr<ActionVector> sparse_action_set = make_unique<ActionVector>();
        double i = 0;
        double sum_values = 0.0;
        for (auto it=qval_to_act_for_search.rbegin(); it != qval_to_act_for_search.rend(); it++) {
            double value = it->first;
            if (!std::isfinite(value)) continue;
            shared_ptr<const Action> action = it->second;
            if (only_actions_with_children && !has_child_node(action)) continue;
            sum_values += value;
            if (1.0 + (i+1.0)*value > sum_values) {
                sparse_action_set->push_back(action);
            }
            i++;
        }
        return sparse_action_set;
    }

    /**
     * Computes the spmax
     * http://proceedings.mlr.press/v139/dam21a/dam21a.pdf
     * This just computes the spmax equation given in the paper
    */
    double TentsDNode::spmax(bool for_backup, bool only_actions_with_children) const {
        unique_ptr<ActionVector> sparse_action_set = get_sparse_action_set(for_backup, only_actions_with_children);

        double sum_sparse_values = 0.0;
        for (shared_ptr<const Action> action : *sparse_action_set) {
                sum_sparse_values += act_to_qval.at(action);
        }

        size_t sparse_size = sparse_action_set->size();
        if (sparse_size == 0) {
            return 0.5;
        }
        double spmax_common_term = 0.5 * pow(sum_sparse_values-1.0, 2.0) / pow(static_cast<double>(sparse_size), 2.0);
        double spmax = 0.5;
        for (shared_ptr<const Action> action : *sparse_action_set) {
            double action_val = act_to_qval.at(action);
            spmax += pow(action_val, 2.0) / 2.0 - spmax_common_term;
        }

        return spmax;
    }

    /**
     * Compute action weights.
     * 
     * Computes the distribution according to the paper:
     * http://proceedings.mlr.press/v139/dam21a/dam21a.pdf
     * 
     * N.B. ments locks children around calling this, so have lock on children
     * 
     * TODO: add normalising q values to range [0,1] (as in ments)
     */
    void TentsDNode::compute_action_weights(
        ActionDistr& action_weights, 
        double& sum_action_weights, 
        double& normalisation_term, 
        ThtsContext& context,
        bool for_backup,
        bool only_actions_with_children) const
    {
        sum_action_weights = 0.0;
        normalisation_term = 0.0;

        // compute the common term (guard empty sparse set to avoid division by zero → NaN)
        unique_ptr<ActionVector> sparse_action_set = get_sparse_action_set(for_backup,only_actions_with_children);
        if (sparse_action_set->empty()) {
            size_t n_actions = actions->size();
            if (n_actions == 0) {
                return;
            }
            for (shared_ptr<const Action> action : *actions) {
                action_weights[action] = 1.0;
            }
            sum_action_weights = n_actions;
            return;
        }

        // Get q values and normalise them
        for (shared_ptr<const Action> action : *actions) {
            double q_value = get_soft_q_value_safe(action, for_backup);
            if (!std::isfinite(q_value)) continue;
            action_weights[action] = q_value;
        }

        MentsManager& manager = (MentsManager&) *thts_manager;
        if (!for_backup && manager.normalise_q_values) 
        {
            thts::helper::linearly_normalise_values(action_weights);
        }

        // Compute common term
        double sum_sparse_values = 0.0;
        double temp = get_temp_safe();
        for (shared_ptr<const Action> action : *sparse_action_set) {
            double q_over_t = action_weights[action] / temp;
            sum_sparse_values += q_over_t;
        }
        double common_term = 0.0;
        if (sparse_action_set->size() > 0) 
        {
            common_term = (sum_sparse_values - 1.0) / static_cast<double>(sparse_action_set->size());
        }

        // compute weights and store (skip non-finite to avoid propagating NaN)
        sum_action_weights = 0.0;
        for (shared_ptr<const Action> action : *actions) {
            double q_over_t = action_weights[action] / temp;
            if (!std::isfinite(q_over_t)) 
            {
                action_weights.erase(action);
                continue;
            }
            double weight = q_over_t - common_term;
            if (weight < 0.0) weight = 0.0;
            action_weights[action] = weight;
            sum_action_weights += weight;
        }
        
        // If all action weights extremely small or none valid, fall back to uniform for numerical stability
        if (sum_action_weights < EPS) {
            size_t n_actions = actions->size();
            if (n_actions == 0) {
                sum_action_weights = 1.0;
                return;
            }
            double uniform_weight = 1.0 / static_cast<double>(n_actions);
            for (shared_ptr<const Action> action : *actions) {
                action_weights[action] = uniform_weight;
            }
            sum_action_weights = 1.0;
        }
    }

    /**
     * Calls the ments implementation of select action and stores the action in the context at 
     * "{decision_depth}" -> selected_action
     */
    shared_ptr<const Action> TentsDNode::select_action(ThtsContext& ctx) {
        shared_ptr<const Action> selected_action = select_action_ments(ctx);
        return selected_action;
    }

    /**
     * Get action from context
     * Get q_value (possibly from child)
     * Update value in map
    */
   void TentsDNode::backup_update_map(ThtsContext& ctx) {
        for (shared_ptr<const Action> action : *actions) 
        {
            bool for_backup = false;
            double new_q_value = get_soft_q_value_over_temp_safe(action, for_backup);
            update_maps(action, new_q_value, for_backup);
            for_backup = true;
            new_q_value = get_soft_q_value_over_temp_safe(action, for_backup);
            update_maps(action, new_q_value, for_backup);
        }
   }

    /**
     * Perform tents backup
     * I.e. soft_value = temp * spmax(), remembering that if we are an opponent, we negated the Q(s,a) values, and need 
     * to negate again, so that values are stored w.r.t. the first player.
     * And remember to increment number of backups!
    */
   void TentsDNode::backup_tents(ThtsContext& ctx) {
        num_backups++;

        backup_update_map(ctx);

        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        double temp = get_temp();
        bool for_backup = true;
        bool only_actions_with_children = true;
        soft_value = opp_coeff * temp * spmax(for_backup, only_actions_with_children);
        soft_value_local = soft_value;

        if (has_heuristic_value()) 
        {
            double effective_num_backups = num_backups + thts_manager->heuristic_weight_global;
            soft_value *= num_backups / effective_num_backups;
            soft_value += thts_manager->heuristic_weight_global * heuristic_value / effective_num_backups;

            effective_num_backups += thts_manager->heuristic_weight_local;
            soft_value_local *= num_backups / effective_num_backups;
            soft_value_local += thts_manager->heuristic_weight_local * heuristic_value / effective_num_backups;
        }
   }

    /**
     * Calls the ments implementation of backup, performing soft backup
     */
    void TentsDNode::backup(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsContext& ctx) 
    {
        backup_tents(ctx);
    }

    /**
     * Make child node
     */
    shared_ptr<TentsCNode> TentsDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<TentsCNode>(
            static_pointer_cast<MentsManager>(thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const TentsDNode>(shared_from_this()));
    }
}



/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {

    shared_ptr<ThtsCNode> TentsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<TentsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}
