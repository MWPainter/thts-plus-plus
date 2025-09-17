
#include "mo/mo_thts_decision_node.h"

#include "mo/mo_thts_manager.h"

#include <stdexcept>
#include <tuple>
#include <utility>

using namespace std;


namespace thts {
    /**
     * Constructor mostly uses initialisation list. 
     * 
     * Nuance use of heuristic value is to enforce nodes for sink states to have a value of zero
     */
    MoThtsDNode::MoThtsDNode(
        shared_ptr<MoThtsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const MoThtsCNode> parent) :
            ThtsDNode(thts_manager, state, decision_depth, decision_timestep, parent),
            mo_heuristic_value(thts_manager->reward_dim, 0.0),
            vector_visit_count(thts_manager->reward_dim, 0.0),
            local_backups(0),
            total_cnode_backups_in_subtree(0),
            total_dnode_backups_in_subtree(0)
    {
        if (thts_manager->mo_heuristic_fn != nullptr
            && !thts_manager->thts_env()->is_sink_state_itfc(state, *thts_manager->get_thts_context()))
        {
            MoThtsEnv& mo_thts_env = (MoThtsEnv&) *dynamic_pointer_cast<MoThtsEnv>(thts_manager->thts_env());
            mo_heuristic_value = thts_manager->mo_heuristic_fn(state, mo_thts_env, *thts_manager, decision_depth);
        }
    }

    void MoThtsDNode::visit_itfc(ThtsContext& ctx) {
        ThtsDNode::visit_itfc(ctx);
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        vector_visit_count += mo_ctx.context_weight;
    }

    double MoThtsDNode::get_num_visits(MoThtsContext& ctx) const {
        MoThtsManager& mo_thts_manager = (MoThtsManager&) *thts_manager;
        if (mo_thts_manager.use_vector_visit_counts) {
            return vector_visit_count.dot(ctx.context_weight);
        }
        return num_visits;
    }

    double MoThtsDNode::get_scalar_num_visits() const {
        return num_visits;
    }

    Vec MoThtsDNode::get_vector_num_visits() const {
        return vector_visit_count;
    }

    /**
     * Raise error if call wrong backup fn
    */
    void MoThtsDNode::backup_itfc(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsContext& ctx) 
    {
        throw runtime_error("Called single objective backup function for multi objective node");
    }

    /**
     * Get an (approximate) convex hull from this node
     */
    ConvexHull MoThtsDNode::get_convex_hull() const 
    {
        throw runtime_error("Calling MoThtsDNode::get_convex_hull, has this been overriden in algorithm trying to use?");
    }

    void MoThtsDNode::increment_and_update_backup_count() {
        local_backups++;

        total_cnode_backups_in_subtree = 0;
        total_dnode_backups_in_subtree = local_backups;
        for (pair<shared_ptr<const Action>,shared_ptr<ThtsCNode>> pair : children) {
            MoThtsCNode& child = (MoThtsCNode&) *pair.second;
            total_cnode_backups_in_subtree += child.total_cnode_backups_in_subtree;
            total_dnode_backups_in_subtree += child.total_dnode_backups_in_subtree;
        }
    }

    int MoThtsDNode::get_total_backups_in_subtree() {
        return total_cnode_backups_in_subtree + total_dnode_backups_in_subtree;
    }

    int MoThtsDNode::get_cnode_backups_in_subtree() {
        return total_cnode_backups_in_subtree;
    }

    int MoThtsDNode::get_dnode_backups_in_subtree() {
        return total_dnode_backups_in_subtree;
    }
}