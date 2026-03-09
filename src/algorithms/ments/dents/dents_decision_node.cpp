#include "algorithms/ments/dents/dents_decision_node.h"

#include "algorithms/common/decaying_temp.h"

using namespace std; 

static double EPS = 1e-16;

namespace thts {
    DentsDNode::DentsDNode(
        shared_ptr<DentsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const DentsCNode> parent) :
            DBMentsDNode(
                static_pointer_cast<MentsManager>(thts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const DBMentsCNode>(parent)),
            EntDNode(),
            EmpNode(1, heuristic_value)
    {
    }

    /**
     * Get decayed temp
     */
    double DentsDNode::get_entropy_coeff() const {
        DentsManager& manager = (DentsManager&) *thts_manager;
        Schedule& entropy_coeff_schedule = *manager.entropy_coeff_schedule_ptr;
        return entropy_coeff_schedule(num_visits);
    }

    /**
     * Get the raw q value of a child node
     */
    double DentsDNode::get_q_value(std::shared_ptr<const Action> action, double opp_coeff, bool for_backup) const {
        if (!has_child_node(action)) {
            // returns the heuristic value
            return MentsDNode::get_soft_q_value(action, opp_coeff, for_backup);
        }

        DentsManager& manager = (DentsManager&) *thts_manager;
        DentsCNode& child = (DentsCNode&) *get_child_node(action);
        if (!manager.use_dp_value) 
        {
            return child.avg_return;
        }
        if (!for_backup) {
            return child.dp_value_local;
        }
        return child.dp_value;
    }

    /**
     * Get the raw entropy q value of a child node
     */
    double DentsDNode::get_entropy_q_value_term(std::shared_ptr<const Action> action) const {
        if (!has_child_node(action)) {
            return 0.0;
        }

        DentsManager& manager = (DentsManager&) *thts_manager;
        DentsCNode& child = (DentsCNode&) *get_child_node(action);
        return child.subtree_entropy;
    }

    /**
     * In DENTS, soft q values are computed as V + value_temp * H, where V is the dp value of the child,
     * H is the subtree entropy of the child and value_temp is the decayed temperature to weight entropy.
     *
     * To make value temp scale insensitive, we optionally normalise the q values and entropies before 
     * combinding them into a soft q value
     *
     * Old docstring for get_soft_q_value (now unused):
     * Gets the soft q value of a child node (as considered by this current node).
     * 
     * These values are of the form V + temp_decayed * H. Note that the 'temp_decayed' used is the decayed temperature 
     * for *this* node, not the child node, and the soft value returned != child.soft_value.
     * 
     * Other cases are suitably handled by the implementation in MentsDNode, so just call that
     */
    void DentsDNode::fill_soft_q_values(
        unordered_map<shared_ptr<const Action>,double>& soft_q_values,
        double opp_coeff,
        bool for_backup) const
    {
        DentsManager& manager = (DentsManager&) *ThtsDNode::thts_manager;

        // Get current q values
        unordered_map<shared_ptr<const Action>,double> q_values;
        for (shared_ptr<const Action> action : *actions) {
            q_values[action] = get_q_value(action, opp_coeff, for_backup);
        }

        // Normalise Q values
        if (!for_backup && manager.normalise_entropy_before_adding) {
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

        // Get entropy terms
        unordered_map<shared_ptr<const Action>,double> entropy_terms;
        for (shared_ptr<const Action> action : *actions) {
            entropy_terms[action] = get_entropy_q_value_term(action);
        }

        // Normalise entropy terms
        if (!for_backup && manager.normalise_entropy_before_adding) {
            double min_entropy_term = numeric_limits<double>::max();
            double max_entropy_term = numeric_limits<double>::lowest();
            for (pair<shared_ptr<const Action>,double> pr : entropy_terms) {
                double entropy_term = pr.second;
                if (entropy_term < min_entropy_term) min_entropy_term = entropy_term;
                if (entropy_term > max_entropy_term) max_entropy_term = entropy_term;
            }
            for (pair<shared_ptr<const Action>,double> pr : entropy_terms) {
                shared_ptr<const Action> action = pr.first;
                double entropy_term = pr.second;
                entropy_terms[action] = (entropy_term - min_entropy_term) / (max_entropy_term - min_entropy_term + EPS);
            }
        }

        // Combine q values and entropy terms to fill in soft q values
        double entropy_coeff = get_entropy_coeff();
        for (shared_ptr<const Action> action : *actions) {
            soft_q_values[action] = q_values[action] + entropy_coeff * entropy_terms[action] * entropy_coeff;
        }
    } 

    /**
     * Calls the empnode implementation of recommend action
     */
    shared_ptr<const Action> DentsDNode::recommend_action_best_empirical_value() const {
        if (children.size() == 0u) {
            int index = thts_manager->get_rand_int(0, actions->size());
            return actions->at(index);
        }

        DentsManager& manager = (DentsManager&) *ThtsDNode::thts_manager;
        return EmpNode::recommend_action_best_emp_value<DentsCNode>(
            children, *thts_manager, manager.recommend_visit_threshold, is_opponent());
    }

    /**
     * Implements recommend action to call best dp value
    */
    shared_ptr<const Action> DentsDNode::recommend_action(ThtsContext& ctx) const {
        DentsManager& manager = (DentsManager&) *thts_manager;
        if (manager.recommend_most_visited) {
            return recommend_action_most_visited();
        }
        if (manager.use_dp_value) {
            return recommend_action_best_dp_value();
        }
        return recommend_action_best_empirical_value();
    }

    /**
     * Calls both the entropy backup and dp backup
     * 
     * Recall that the dp backup needs to be passed the type of the child nodes (so can keep dp logic in dp node)
     */
    void DentsDNode::backup(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsContext& ctx) 
    {
        MentsDNode::num_backups++;

        // entropy backup
        ActionDistr action_distr;
        compute_action_distribution(action_distr, ctx);
        backup_ent<DentsCNode>(children, action_distr, is_opponent());

        // value backup
        double val_estimate;
        double val_estimate_local;
        DentsManager& manager = (DentsManager&) *thts_manager;
        if (manager.use_dp_value) {
            backup_dp<DentsCNode>(
                children, 
                has_heuristic_value(), 
                thts_manager->heuristic_weight_global,
                thts_manager->heuristic_weight_local,
                heuristic_value, 
                is_opponent());
            val_estimate = dp_value;
            val_estimate_local = dp_value_local;
        } else {
            backup_emp(trial_cumulative_return_after_node);
            val_estimate = avg_return;
            val_estimate_local = avg_return;
        }
    
        // update local soft_value so that value is sensible / for pretty printing
        // N.B. not actually used in algo
        soft_value = val_estimate + get_entropy_coeff() * subtree_entropy;
        soft_value_local = val_estimate_local + get_entropy_coeff() * subtree_entropy;
    }

    /**
     * Make child node
     */
    shared_ptr<DentsCNode> DentsDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<DentsCNode>(
            static_pointer_cast<DentsManager>(ThtsDNode::thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const DentsDNode>(shared_from_this()));
    }

    /**
     * Return string with all of the relevant values in this node
     */
    string DentsDNode::get_pretty_print_val() const {
        DentsManager& manager = (DentsManager&) *thts_manager;
        double val_estimate = manager.use_dp_value ? dp_value : avg_return; 

        stringstream ss;
        ss << val_estimate << "(temp:" << get_temp() << ",entrpy:" << subtree_entropy << ",val_temp:" << get_entropy_coeff() 
            << ",soft_val:" << soft_value << ")";
        return ss.str();
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    shared_ptr<ThtsCNode> DentsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<DentsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}
