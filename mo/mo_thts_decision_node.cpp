
#include "mo/mo_thts_decision_node.h"

#include "mo/mo_thts_manager.h"

#include <cmath>
#include <limits>
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
        shared_ptr<const MoThtsCNode> parent,
        bool eval_mo_heuristic) :
            ThtsDNode(thts_manager, state, decision_depth, decision_timestep, parent),
            mo_heuristic_value(thts_manager->reward_dim, 0.0),
            vector_visit_count(thts_manager->reward_dim, 0.0),
            local_backups(0),
            total_cnode_backups_in_subtree(0),
            total_dnode_backups_in_subtree(0),
            solved_labelling(0),
            solved_labelling_confidence_interval_range(numeric_limits<double>::max())
    {
        if (eval_mo_heuristic && thts_manager->mo_heuristic_fn != nullptr
            && !thts_manager->thts_env()->is_sink_state_itfc(state, *thts_manager->get_thts_context()))
        {
            MoThtsEnv& mo_thts_env = (MoThtsEnv&) *dynamic_pointer_cast<MoThtsEnv>(thts_manager->thts_env());
            mo_heuristic_value = thts_manager->mo_heuristic_fn(state, mo_thts_env, *thts_manager, decision_depth);
            vector_visit_count = Vec(thts_manager->reward_dim, thts_manager->heuristic_psuedo_trials);
            num_visits = thts_manager->heuristic_psuedo_trials;
            local_backups = thts_manager->heuristic_psuedo_trials;
        }
    }

    vector<shared_ptr<const Action>> MoThtsDNode::get_actions_to_consider() const {
        vector<shared_ptr<const Action>> actions_to_consider;
        MoThtsManager& mo_thts_manager = (MoThtsManager&) *thts_manager;
        
        if (!mo_thts_manager.use_solved_labelling) {
            // Return all children
            for (const auto& pair : children) {
                actions_to_consider.push_back(pair.first);
            }
            return actions_to_consider;
        }
        
        // Find the minimum solved labelling among children
        int min_solved_labelling = numeric_limits<int>::max();
        for (const auto& pair : children) {
            MoThtsCNode& child = (MoThtsCNode&) *pair.second;
            int child_solved_labelling = child.get_solved_labelling();
            if (child_solved_labelling < min_solved_labelling) {
                min_solved_labelling = child_solved_labelling;
            }
        }
        
        // Return only children with the minimum solved labelling
        for (const auto& pair : children) {
            MoThtsCNode& child = (MoThtsCNode&) *pair.second;
            if (child.get_solved_labelling() == min_solved_labelling) {
                actions_to_consider.push_back(pair.first);
            }
        }
        
        return actions_to_consider;
    }

    int MoThtsDNode::get_local_solved_labelling() const {
        MoThtsManager& mo_thts_manager = (MoThtsManager&) *thts_manager;
        double delta = get_local_solved_labelling_confidence_interval_range();
        double tau = mo_thts_manager.solved_labelling_tolerance;
        
        // Return 0 if not solved (delta > tau)
        if (delta > tau) {
            return 0;
        }
        
        // Find the minimum i such that delta > tau / 2^i
        // Equivalently, the maximum i such that delta <= tau / 2^i
        // i.e., 2^i <= tau / delta
        // i.e., i <= log2(tau / delta)
        int i = static_cast<int>(floor(log2(tau / delta)));
        return max(1, i);
    }

    int MoThtsDNode::get_solved_labelling() const {
        return solved_labelling;
    }

    double MoThtsDNode::get_solved_labelling_confidence_interval_range() const {
        int solved_labelling = get_solved_labelling();
        if (solved_labelling == 0) {
            MoThtsManager& mo_thts_manager = (MoThtsManager&) *thts_manager;
            return mo_thts_manager.solved_labelling_value_scaling;
        }
        return solved_labelling_confidence_interval_range;
    }

    double MoThtsDNode::get_local_solved_labelling_confidence_interval_range() const {
        return solved_labelling_confidence_interval_range;
    }

    void MoThtsDNode::update_solved_labelling() {
        int min_labelling = get_local_solved_labelling();
        
        for (const auto& pair : children) {
            MoThtsCNode& child = (MoThtsCNode&) *pair.second;
            int child_labelling = child.get_solved_labelling();
            if (child_labelling < min_labelling) {
                min_labelling = child_labelling;
            }
        }
        
        solved_labelling = min_labelling;

        update_solved_labelling_confidence_interval_range();
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