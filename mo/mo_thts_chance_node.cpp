#include "mo/mo_thts_chance_node.h"

using namespace std;


namespace thts {
    /**
     * Constructor mostly uses initialisation list. 
     */
    MoThtsCNode::MoThtsCNode(
        shared_ptr<MoThtsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const MoThtsDNode> parent) :
            ThtsCNode(thts_manager, state, action, decision_depth, decision_timestep, parent),
            vector_visit_count(thts_manager->reward_dim),
            local_backups(0),
            total_cnode_backups_in_subtree(0),
            total_dnode_backups_in_subtree(0)
    {
    }

    void MoThtsCNode::visit_itfc(ThtsContext& ctx) {
        ThtsCNode::visit_itfc(ctx);
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        vector_visit_count += mo_ctx.context_weight;
    }

    double MoThtsCNode::get_num_visits(MoThtsContext& ctx) const {
        MoThtsManager& mo_thts_manager = (MoThtsManager&) *thts_manager;
        if (mo_thts_manager.use_vector_visit_counts) {
            return vector_visit_count.dot(ctx.context_weight);
        }
        return num_visits;
    }

    double MoThtsCNode::get_scalar_num_visits() const {
        return num_visits;
    }

    Vec MoThtsCNode::get_vector_num_visits() const {
        return vector_visit_count;
    }

    /**
     * Raise error if call wrong backup fn
    */
    void MoThtsCNode::backup_itfc(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsContext& ctx) 
    {
        throw runtime_error("Called single objective backup function for multi objective node");
    }

    void MoThtsCNode::increment_and_update_backup_count() {
        local_backups++;
        
        total_cnode_backups_in_subtree = local_backups;
        total_dnode_backups_in_subtree = 0;
        for (pair<shared_ptr<const Observation>,shared_ptr<ThtsDNode>> pair : children) {
            MoThtsDNode& child = (MoThtsDNode&) *pair.second;
            total_cnode_backups_in_subtree += child.total_cnode_backups_in_subtree;
            total_dnode_backups_in_subtree += child.total_dnode_backups_in_subtree;
        }
    }

    int MoThtsCNode::get_total_backups_in_subtree() {
        return total_cnode_backups_in_subtree + total_dnode_backups_in_subtree;
    }

    int MoThtsCNode::get_cnode_backups_in_subtree() {
        return total_cnode_backups_in_subtree;
    }

    int MoThtsCNode::get_dnode_backups_in_subtree() {
        return total_dnode_backups_in_subtree;
    }
}