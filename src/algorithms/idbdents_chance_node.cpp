#include "algorithms/idbdents_chance_node.h"

using namespace std;

namespace thts {
    /**
     * Constructor
     */
    IDBDentsCNode::IDBDentsCNode(
        shared_ptr<IDentsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const IDBDentsDNode> parent) :
            DBMentsCNode(
                static_pointer_cast<MentsManager>(thts_manager),
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const DBMentsDNode>(parent)),
            subtree_entropy(0.0)
    {
    }

    /**
     * Entropy = expected value of child entropies (i.e. empirical average)
     * 
     * Adapted from DPDNode DPBackup function
    */
    void IDBDentsCNode::backup_entropy() {
        lock_all_children();
        subtree_entropy = 0.0;
        double sum_child_backups = 0;
        for (pair<shared_ptr<const Observation>,shared_ptr<ThtsDNode>> pr : children) {
            IDBDentsDNode& child = (IDBDentsDNode&) *pr.second;
            int child_backups = child.MentsDNode::num_backups;
            if (child_backups == 0) continue;
            sum_child_backups += child_backups;
            subtree_entropy *= (sum_child_backups - child_backups) / sum_child_backups;
            subtree_entropy += child_backups * child.subtree_entropy / sum_child_backups; 
        }
        unlock_all_children();
    }

    /**
     * Calls entropy backup and dp backup
     * 
     * Recall that the dp backup needs to be passed the type of the child nodes (so can keep dp logic in dp node)
     */
    void IDBDentsCNode::backup(
        const std::vector<double>& trial_rewards_before_node, 
        const std::vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsEnvContext& ctx)
    {   
        // un-needed technically, but call to get a value for soft_value
        // backup_soft();
        MentsCNode::num_backups++;

        // actually needed backups
        backup_entropy();
        backup_dp<DBMentsDNode>(children, local_reward, is_opponent());
    }

    /**
     * Make child node
     */
    shared_ptr<IDBDentsDNode> IDBDentsCNode::create_child_node_helper(
        shared_ptr<const State> observation, shared_ptr<const State> next_state) const 
    {  
        shared_ptr<const State> mdp_next_state = static_pointer_cast<const State>(observation);
        return make_shared<IDBDentsDNode>(
            static_pointer_cast<IDentsManager>(ThtsCNode::thts_manager), 
            mdp_next_state,
            decision_depth+1, 
            decision_timestep+1, 
            static_pointer_cast<const IDBDentsCNode>(shared_from_this()));
    }

    /**
     * Return string with all of the relevant values in this node
     */
    string IDBDentsCNode::get_pretty_print_val() const {
        stringstream ss;
        ss << dp_value << "(s:" << soft_value << ",e:" << subtree_entropy << ")";
        return ss.str();
    }
}

/**
 * Boilerplate ThtsCNode interface implementation. Copied from thts_chance_node_template.h.
 */
namespace thts {
    shared_ptr<ThtsDNode> IDBDentsCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation, shared_ptr<const State> next_state) const 
    {
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<IDBDentsDNode> child_node = create_child_node_helper(obsv_itfc, next_state_itfc);
        return static_pointer_cast<ThtsDNode>(child_node);
    }
}