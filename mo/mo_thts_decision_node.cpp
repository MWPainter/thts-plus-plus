
#include "mo/mo_thts_decision_node.h"

#include "mo/mo_thts_manager.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <iostream>

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
            solved_value(1.0)
    {
        bool is_sink = thts_manager->thts_env()->is_sink_state_itfc(state, *thts_manager->get_thts_context());
        
        if (is_sink) 
        {
            solved_value = 0.0;
        }

        if (eval_mo_heuristic && thts_manager->mo_heuristic_fn != nullptr && !is_sink)
        {
            MoThtsEnv& mo_thts_env = (MoThtsEnv&) *dynamic_pointer_cast<MoThtsEnv>(thts_manager->thts_env());
            mo_heuristic_value = thts_manager->mo_heuristic_fn(state, mo_thts_env, *thts_manager, decision_depth);
            vector_visit_count = Vec(thts_manager->reward_dim, thts_manager->heuristic_psuedo_trials);
            num_visits = thts_manager->heuristic_psuedo_trials;
            local_backups = thts_manager->heuristic_psuedo_trials;
        }
    }

    vector<shared_ptr<const Action>> MoThtsDNode::get_actions_to_consider(ThtsContext& ctx) const {
        MoThtsManager& mo_thts_manager = static_cast<MoThtsManager&>(*thts_manager);
        ThtsEnv& thts_env = *mo_thts_manager.thts_env();
        shared_ptr<ActionVector> all_actions = thts_env.get_valid_actions_itfc(this->state, ctx);

        // If not using solved labelling, return all actions
        if (!mo_thts_manager.use_solved_labelling) {
            return *all_actions;
        }
        
        // If num children < num actions, then we are unsolved
        // Return the unexpanded actions + unsolved children
        vector<shared_ptr<const Action>> actions_to_consider;
        if (children.size() < all_actions->size()) 
        {
            for (const auto& action : *all_actions) {
                if (!has_child_node_itfc(action)) {
                    actions_to_consider.push_back(action);
                }
            }
            for (const auto& pair : children) {
                MoThtsCNode& child = (MoThtsCNode&) *pair.second;
                int child_solved_level = child.get_solved_level();
                if (child_solved_level == 0) {
                    actions_to_consider.push_back(pair.first);
                }
            }
            return actions_to_consider;
        }

        // If we get here, there is a child for every action
        // Find the minimum solved labelling among children
        int min_solved_level = numeric_limits<int>::max();
        for (const auto& pair : children) {
            MoThtsCNode& child = (MoThtsCNode&) *pair.second;
            int child_solved_level = child.get_solved_level();
            if (child_solved_level < min_solved_level) {
                min_solved_level = child_solved_level;
            }
        }
        
        // Return only children/actions with the minimum solved level
        for (const auto& pair : children) {
            MoThtsCNode& child = (MoThtsCNode&) *pair.second;
            if (child.get_solved_level() == min_solved_level) {
                actions_to_consider.push_back(pair.first);
            }
        }
        
        return actions_to_consider;
    }

    int MoThtsDNode::get_solved_level() const {
        MoThtsManager& mo_thts_manager = (MoThtsManager&) *thts_manager;
        double delta = get_solved_value();
        double tau = mo_thts_manager.solved_labelling_tolerance;
        
        // Return 0 if not solved (delta > tau)
        if (delta > tau) {
            return 0;
        }

        // If solved, return max level
        if (delta == 0) {
            return std::numeric_limits<int>::max();
        }
        
        // Find the minimum i such that delta > tau / 2^(i-1)
        // Equivalently, the maximum i such that delta <= tau / 2^(i-1)
        // i.e., 2^(i-1) <= tau / delta
        // i.e., i-1 <= log2(tau / delta)
        // i.e., i <= 1 + log2(tau / delta)
        double log_ratio = log2(tau / delta);
        if (log_ratio >= std::numeric_limits<int>::max() - 1) {
            return std::numeric_limits<int>::max();
        }
        return static_cast<int>(floor(1 + log_ratio));
    }

    double MoThtsDNode::get_solved_value() const 
    {
        return this->solved_value;
    }

    void MoThtsDNode::update_solved_value() 
    {
        // If we are a sink node, we are solved
        if (is_sink()) {
            this->solved_value = 0.0;
            return;
        }

        // If there are any actions we have never taken, we are unsolved
        size_t num_actions = thts_manager->thts_env()->get_valid_actions_itfc(
            state, *thts_manager->get_thts_context())->size();
        if (num_actions > children.size()) {
            this->solved_value = 1.0;
            return;
        }

        // Otherwise, we take the maximum solved value of our children
        double max_solved_value = 0.0;
        for (const auto& pair : children) {
            MoThtsCNode& child = (MoThtsCNode&) *pair.second;
            double child_solved_value = child.get_solved_value();
            if (child_solved_value > max_solved_value) {
                max_solved_value = child_solved_value;
            }
        }
        this->solved_value = max_solved_value;
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