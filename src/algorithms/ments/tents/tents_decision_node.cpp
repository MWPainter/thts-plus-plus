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
        for (shared_ptr<const Action> action : *actions) {
            double qval = get_soft_q_value_over_temp(action, false);
            qval_to_act.insert(make_pair(qval, action));
            act_to_qval.insert_or_assign(action, qval);
            double qval_for_search = get_soft_q_value_over_temp(action, true);
            qval_to_act_for_search.insert(make_pair(qval_for_search, action));
            act_to_qval_for_search.insert_or_assign(action, qval_for_search);
        }

        stringstream ss;
        ss << decision_depth;
        _selected_action_key = ss.str();
    }

    /**
     * Get the value of Q(s,a)/temp from the best available source (see ments get_soft_q_value, tries child, then prior)
    */
    double TentsDNode::get_soft_q_value_over_temp(shared_ptr<const Action> action, bool for_search) const {
        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        double qval = get_soft_q_value(action, opp_coeff, for_search);
        return qval / get_temp();
    }

    /**
     * Updates the tents mapping for 'action' to/from 'neq_q_value'
    */
    void TentsDNode::update_maps(shared_ptr<const Action> action, double new_q_value, bool for_search) {
        if (!for_search) {
            double old_q_value = act_to_qval[action];
            act_to_qval.erase(action);
            for (auto it=qval_to_act.find(old_q_value); it != qval_to_act.end(); it++) {
                if (it->first != old_q_value) throw runtime_error("Error in updating Tents maps.");
                if (it->second != action) continue;
                qval_to_act.erase(it);
                break;
            }

            act_to_qval.insert_or_assign(action, new_q_value);
            qval_to_act.insert(make_pair(new_q_value, action));

            return;
        }

        double old_q_value_for_search = act_to_qval_for_search[action];
        act_to_qval_for_search.erase(action);
        for (auto it=qval_to_act_for_search.find(old_q_value_for_search); it != qval_to_act_for_search.end(); it++) {
            if (it->first != old_q_value_for_search) throw runtime_error("Error in updating Tents maps.");
            if (it->second != action) continue;
            qval_to_act_for_search.erase(it);
            break;
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
    unique_ptr<ActionVector> TentsDNode::get_sparse_action_set(bool for_search) const {
        if (!for_search) {
            unique_ptr<ActionVector> sparse_action_set = make_unique<ActionVector>();
            double i = 0;
            double sum_values = 0.0;
            for (auto it=qval_to_act.rbegin(); it != qval_to_act.rend(); it++) {
                double value = it->first;
                shared_ptr<const Action> action = it->second;
                sum_values += value;
                if (1.0 + (i+1.0)*value > sum_values) {
                    sparse_action_set->push_back(action);
                }
                i++;
            }
            return sparse_action_set;
        }


        unique_ptr<ActionVector> sparse_action_set = make_unique<ActionVector>();
        double i = 0;
        double sum_values = 0.0;
        for (auto it=qval_to_act_for_search.rbegin(); it != qval_to_act_for_search.rend(); it++) {
            double value = it->first;
            shared_ptr<const Action> action = it->second;
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
    double TentsDNode::spmax(bool for_search) const {
        unique_ptr<ActionVector> sparse_action_set = get_sparse_action_set(for_search);

        double sum_sparse_values = 0.0;
        if (!for_search) {
            for (shared_ptr<const Action> action : *sparse_action_set) {
                    sum_sparse_values += act_to_qval.at(action);
            }
        } else {
            for (shared_ptr<const Action> action : *sparse_action_set) {
                sum_sparse_values += act_to_qval_for_search.at(action);
            }
        }

        double spmax_common_term = 0.5 * pow(sum_sparse_values-1.0, 2.0) / pow(sparse_action_set->size(), 2.0);
        double spmax = 0.5;
        if (!for_search) {
            for (shared_ptr<const Action> action : *sparse_action_set) {
                double action_val = act_to_qval.at(action);
                spmax += pow(action_val, 2.0) / 2.0 - spmax_common_term;
            }
        } else {
            for (shared_ptr<const Action> action : *sparse_action_set) {
                double action_val = act_to_qval_for_search.at(action);
                spmax += pow(action_val, 2.0) / 2.0 - spmax_common_term;
            }
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
        bool for_search) const
    {
        sum_action_weights = 0.0;
        normalisation_term = 0.0;

        // compute the common term
        unique_ptr<ActionVector> sparse_action_set = get_sparse_action_set(for_search);
        double sum_sparse_values = 0.0;
        for (shared_ptr<const Action> action : *sparse_action_set) {
            sum_sparse_values += get_soft_q_value_over_temp(action, for_search);
        }
        double common_term = (sum_sparse_values - 1.0) / sparse_action_set->size();

        // compute weights and store
        for (shared_ptr<const Action> action : *actions) {
            double weight = get_soft_q_value_over_temp(action, for_search) - common_term;
            if (weight < 0.0) weight = 0.0;
            action_weights[action] = weight;
            sum_action_weights += weight;
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
     * Calls the ments implementation of select action and stores the action in the context at 
     * "{decision_depth}" -> selected_action
     */
    shared_ptr<const Action> TentsDNode::select_action(ThtsContext& ctx) {
        shared_ptr<const Action> selected_action = select_action_ments(ctx);
        ctx.put_value_const(_selected_action_key, selected_action);
        return selected_action;
    }

    /**
     * Get action from context
     * Get q_value (possibly from child)
     * Update value in map
    */
   void TentsDNode::backup_update_map(ThtsContext& ctx) {
        shared_ptr<const Action> selected_action = ctx.get_value_ptr_const<Action>(_selected_action_key);
        double new_q_value = get_soft_q_value_over_temp(selected_action, false);
        update_maps(selected_action, new_q_value, false);
        new_q_value = get_soft_q_value_over_temp(selected_action, true);
        update_maps(selected_action, new_q_value, true);
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
        soft_value = opp_coeff * temp * spmax(false);
        soft_value_for_search = opp_coeff * temp * spmax(true);

        if (has_heuristic_value()) 
        {
            soft_value_for_search *= (num_backups - thts_manager->heuristic_weight) / num_backups;
            soft_value_for_search += thts_manager->heuristic_weight * heuristic_value / num_backups;
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
