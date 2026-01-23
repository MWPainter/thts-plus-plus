#include "mo/mo_thts_chance_node.h"

#include <cmath>
#include <limits>

using namespace std;


namespace thts {
    /**
     * Constructor mostly uses initialisation list. 
     * Uses SkipLocalRewardInit tag to avoid calling get_reward_itfc which is deleted for MO envs.
     */
    MoThtsCNode::MoThtsCNode(
        shared_ptr<MoThtsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const MoThtsDNode> parent) :
            ThtsCNode(SkipLocalRewardInit{}, thts_manager, state, action, decision_depth, decision_timestep, parent),
            vector_visit_count(thts_manager->reward_dim),
            local_backups(0),
            total_cnode_backups_in_subtree(0),
            total_dnode_backups_in_subtree(0),
            solved_labelling(0)
    {
    }

    int MoThtsCNode::get_solved_labelling() const {
        MoThtsManager& mo_thts_manager = (MoThtsManager&) *thts_manager;
        double delta = get_solved_labelling_confidence_interval_range();
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

    double MoThtsCNode::get_solved_labelling_confidence_interval_range() const {
        MoThtsManager& mo_thts_manager = (MoThtsManager&) *thts_manager;
        double delta = mo_thts_manager.solved_labelling_delta_fail_probability;
        double r_max = mo_thts_manager.solved_labelling_value_scaling;
        double n = static_cast<double>(num_visits);
        
        // If no visits yet, return maximum range
        if (n <= 0) {
            return r_max;
        }
        
        // DKW epsilon term: 2 sqrt(log(2/delta) / (2 * n))
        double dkw_epsilon = 2.0 * sqrt(log(2.0 / delta) / (2.0 * n));
        
        // Part 1 & 2: Compute r_seen
        // r_seen = sum_x_seen [r(x) * (q(x) + dkw_epsilon)]
        double r_seen = 0.0;
        int singletons = 0;  // count of outcomes seen exactly once (for Good Turing)
        
        for (const auto& pair : children) {
            MoThtsDNode& child = (MoThtsDNode&) *pair.second;
            double r_x = child.get_solved_labelling_confidence_interval_range();
            
            // Get empirical probability q(x) from empirical_distribution
            auto it = empirical_distribution.find(pair.first);
            int count = (it != empirical_distribution.end()) ? it->second : 0;
            double q_x = static_cast<double>(count) / n;
            
            // r(x) * (q(x) + dkw_epsilon)
            r_seen += r_x * (q_x + dkw_epsilon);
            
            // Count singletons for Good Turing estimate
            if (count == 1) {
                singletons++;
            }
        }
        
        // Part 3: Compute r_missing using Good Turing estimate
        // M' = c/n where c is the number of singletons
        // r_missing = r_max * (M' + sqrt(log(1/delta) / n))
        double M_prime = static_cast<double>(singletons) / n;
        double missing_mass_bound = M_prime + sqrt(log(1.0 / delta) / n);
        double r_missing = r_max * missing_mass_bound;
        
        // Final confidence interval range: r = r_seen + r_missing
        return r_seen + r_missing;
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