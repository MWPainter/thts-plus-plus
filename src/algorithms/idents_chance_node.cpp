#include "algorithms/idents_chance_node.h"

using namespace std;

namespace thts {
    /**
     * Constructor
     */
    IDentsCNode::IDentsCNode(
        shared_ptr<IDentsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const IDentsDNode> parent) :
            MentsCNode(
                thts_manager,
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MentsDNode>(parent)),
            ments_subtree_entropy(0.0),
            subtree_entropy(0.0)
    {
    }

    /**
     * Make child node
     */
    shared_ptr<IDentsDNode> IDentsCNode::create_child_node_helper(
        shared_ptr<const State> observation, shared_ptr<const State> next_state) const 
    {  
        shared_ptr<const State> mdp_next_state = static_pointer_cast<const State>(observation);
        return make_shared<IDentsDNode>(
            static_pointer_cast<IDentsManager>(thts_manager), 
            mdp_next_state,
            decision_depth+1, 
            decision_timestep+1, 
            static_pointer_cast<const IDentsCNode>(shared_from_this()));
    }

    /**
     * Entropy = expected value of child entropies (i.e. empirical average)
     * 
     * Adapted from DPDNode DPBackup function
    */
    void IDentsCNode::backup_entropy() {
        lock_all_children();
        ments_subtree_entropy = 0.0;
        subtree_entropy = 0.0;
        double sum_child_backups = 0;
        for (pair<shared_ptr<const Observation>,shared_ptr<ThtsDNode>> pr : children) {
            IDentsDNode& child = (IDentsDNode&) *pr.second;
            if (child.num_backups == 0) continue;
            sum_child_backups += child.num_backups;
            ments_subtree_entropy *= (sum_child_backups - child.num_backups) / sum_child_backups;
            ments_subtree_entropy += child.num_backups * child.subtree_entropy / sum_child_backups; 
            subtree_entropy *= (sum_child_backups - child.num_backups) / sum_child_backups;
            subtree_entropy += child.num_backups * child.subtree_entropy / sum_child_backups; 
        }
        unlock_all_children();
   }

    /**
     * Calls soft backup and entropy backup
     */
    void IDentsCNode::backup(
        const std::vector<double>& trial_rewards_before_node, 
        const std::vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsEnvContext& ctx)
    {   
        backup_soft();
        backup_entropy();
    }
}

/**
 * Boilerplate ThtsCNode interface implementation. Copied from thts_chance_node_template.h.
 */
namespace thts {
    shared_ptr<ThtsDNode> IDentsCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation, shared_ptr<const State> next_state) const 
    {
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<IDentsDNode> child_node = create_child_node_helper(obsv_itfc, next_state_itfc);
        return static_pointer_cast<ThtsDNode>(child_node);
    }
}