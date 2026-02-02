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
            solved_value(1.0)
    {
    }

    int MoThtsCNode::get_solved_level() const {
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

    double MoThtsCNode::get_solved_value() const 
    {
        return this->solved_value;
    }

    void MoThtsCNode::update_solved_value() const 
    {
        MoThtsManager& mo_thts_manager = (MoThtsManager&) *thts_manager;
        double delta = mo_thts_manager.solved_labelling_delta_fail_probability / 2.0;
        double n = static_cast<double>(num_visits);
        
        // If no visits yet we are unsolved, so return 1.0
        if (n <= 0) {
            this->solved_value = 1.0;
            return;
        }
        
        // DKW epsilon term: 2 sqrt(log(2/delta) / (2 * n))
        double dkw_epsilon = 2.0 * sqrt(log(2.0 / delta) / (2.0 * n));
        
        // Compute e_s_seen, and count singletons for Good Turing estimate
        // e_s_seen = sum_x_seen [s(x) * (q(x) + dkw_epsilon)]
        double e_s_seen = 0.0;
        int singletons = 0;  // count of outcomes seen exactly once (for Good Turing)
        
        for (const auto& pair : children) {
            MoThtsDNode& child = (MoThtsDNode&) *pair.second;
            double s_x = child.get_solved_value();
            
            // Get empirical probability q(x) from empirical_distribution
            auto it = empirical_distribution.find(pair.first);
            int count = (it != empirical_distribution.end()) ? it->second : 0;
            double q_x = static_cast<double>(count) / n;
            
            // r(x) * (q(x) + dkw_epsilon)
            e_s_seen += s_x * (q_x + dkw_epsilon);
            
            // Count singletons for Good Turing estimate
            if (count == 1) {
                singletons++;
            }
        }
        
        // Compute e_s_missing using Good Turing estimate
        // M' = c/n where c is the number of singletons
        // e_s_missing = e_s_max * (M' + sqrt(log(1/delta) / n))
        double M_prime = static_cast<double>(singletons) / n;
        double e_s_missing = M_prime + sqrt(log(1.0 / delta) / n);
        
        // Final solved value set to upper bound: e_bound =e_s_seen + e_s_missing
        // Note that because solved value is in range [0,1], that 1.0 is a trivial upper bound
        this->solved_value = min(e_s_seen + e_s_missing, 1.0);
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