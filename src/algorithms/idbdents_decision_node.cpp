#include "algorithms/idbdents_decision_node.h"

#include "algorithms/common/decaying_temp.h"

using namespace std; 

namespace thts {
    IDBDentsDNode::IDBDentsDNode(
        shared_ptr<IDentsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const IDBDentsCNode> parent) :
            DBMentsDNode(
                static_pointer_cast<MentsManager>(thts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const DBMentsCNode>(parent)),
            local_entropy(0.0),
            subtree_entropy(0.0)
    {
    }

    /**
     * Return the search temp for ments
     */
    double IDBDentsDNode::get_temp() const {
        IDentsManager& manager = (IDentsManager&) *ThtsDNode::thts_manager;
        return manager.search_temp;
    }

    /**
     * CGet decayed temp
     */
    double IDBDentsDNode::get_decayed_temp() const {
        IDentsManager& manager = (IDentsManager&) *ThtsDNode::thts_manager;
        return compute_decayed_temp(manager.temp, num_visits, manager.min_temp);
    }

    /**
     * Gets the soft q value of a child node (as considered by this current node).
     * 
     * These values are of the form V + temp_decayed * H. Note that the 'temp_decayed' used is the decayed temperature 
     * for *this* node, not the child node, and the soft value returned != child.soft_value.
     * 
     * Other cases are suitably handled by the implementation in MentsDNode, so just call that
    */
    double IDBDentsDNode::get_soft_q_value(std::shared_ptr<const Action> action, double opp_coeff) const {
        if (!has_child_node(action)) {
            return MentsDNode::get_soft_q_value(action, opp_coeff);
        }

        IDBDentsCNode& child = (IDBDentsCNode&) *get_child_node(action);
        return opp_coeff * (child.dp_value + get_decayed_temp() * child.subtree_entropy);
    }

    /**
     * Updates the values of the entropies.
     * 
     * Computes the local entropy of the policy at this node
     * 
     * In the two player case, subtree_entropy = entropy_player - entropy_opponent, assuming we have it computed at 
     * subnodes, then we need to do the following:
     * 1. compute expected value of child entropy given the current local policy
     * 2. add local entropy (or subtract if we are the opponent)
     * 
     * N.B. with some maths, we could show H = H_local + sum(Pr(a) * H(a)), where:
     * H = subtree entropy
     * H_local = local entropy
     * Pr(a) = prob select action a
     * H(a) = subtree entropy of child node corresponding to action a
    */
    void IDBDentsDNode::backup_entropy(ThtsEnvContext& ctx) {
        // Compute local entropy (already thread safe)
        ActionDistr action_distr;
        compute_action_distribution(action_distr, ctx);
        local_entropy = 0.0;
        for (pair<shared_ptr<const Action>,double> pr : action_distr) {
            double prob = pr.second;
            local_entropy -= prob * log(prob);
        }

        // Update subtree entropy == sum child subtree entropies + local
        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        subtree_entropy = opp_coeff * local_entropy;
        lock_all_children();
        for (pair<shared_ptr<const Action>,shared_ptr<ThtsCNode>> pr : children) {
            shared_ptr<const Action> action = pr.first;
            IDBDentsCNode& child = (IDBDentsCNode&) *pr.second;
            subtree_entropy += action_distr[action] * child.subtree_entropy;
        }
        unlock_all_children();

        // Remember to update num backups
        MentsDNode::num_backups++;
    }

    /**
     * Calls both the entropy backup and dp backup
     * 
     * Recall that the dp backup needs to be passed the type of the child nodes (so can keep dp logic in dp node)
     */
    void IDBDentsDNode::backup(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsEnvContext& ctx) 
    {
        backup_entropy(ctx);
        backup_dp<DBMentsCNode>(children, is_opponent());

        // update local soft_value so that value is sensible / for pretty printing
        soft_value = dp_value + get_decayed_temp() * subtree_entropy;
    }

    /**
     * Make child node
     */
    shared_ptr<IDBDentsCNode> IDBDentsDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<IDBDentsCNode>(
            static_pointer_cast<IDentsManager>(ThtsDNode::thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const IDBDentsDNode>(shared_from_this()));
    }

    /**
     * Return string with all of the relevant values in this node
     */
    string IDBDentsDNode::get_pretty_print_val() const {
        stringstream ss;
        ss << dp_value << "(s:" << soft_value << ",e:" << subtree_entropy << ",t:" << get_decayed_temp() << ")";
        return ss.str();
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    shared_ptr<ThtsCNode> IDBDentsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<IDBDentsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}