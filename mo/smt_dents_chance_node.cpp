#include "mo/smt_dents_chance_node.h"

using namespace std; 

namespace thts {
    SmtDentsCNode::SmtDentsCNode(
        shared_ptr<SmtDentsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const SmtDentsDNode> parent) :
            SmtBtsCNode(
                static_pointer_cast<SmtBtsManager>(thts_manager),
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const SmtBtsDNode>(parent))
    {
    }

    /**
     * Todo long term: be less wastful with code copy
    */

    /**
     * See comments on NGV datatype for what the pure_backup stuff is about
     */
    void SmtDentsCNode::backup(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        MoThtsContext& ctx) 
    {  
        num_backups++;
        
        // Get closest NGV in simplex map
        shared_ptr<TN> simplex = simplex_map.get_leaf_tn_node(ctx.context_weight);
        shared_ptr<NGV> closest_vertex = simplex->get_closest_ngv_vertex(ctx.context_weight);

        // Compute average value from children
        // And entropy for dents
        SmtDentsManager& manager = (SmtDentsManager&) *thts_manager;
        Eigen::ArrayXd avg_val = Eigen::ArrayXd::Zero(manager.reward_dim);
        bool pure_backup_value_estimate = true;
        double avg_entropy = 0.0;
        int sum_child_backups = 0;
        for (pair<shared_ptr<const Observation>,shared_ptr<ThtsDNode>> pr : children) {
            SmtDentsDNode& child = (SmtDentsDNode&) *pr.second;
            lock_guard<mutex> lg(child.get_lock());
            if (child.num_backups == 0) continue;

            shared_ptr<TN> child_simplex = child.simplex_map.get_leaf_tn_node(closest_vertex->weight);
            shared_ptr<NGV> child_closest_vertex = child_simplex->get_closest_ngv_vertex(closest_vertex->weight);

            sum_child_backups += child.num_backups;
            avg_val *= (sum_child_backups - child.num_backups) / sum_child_backups;
            avg_val += child.num_backups * child_closest_vertex->value_estimate / sum_child_backups; 
            avg_entropy *= (sum_child_backups - child.num_backups) / sum_child_backups;
            avg_entropy += child.num_backups * child_closest_vertex->entropy / sum_child_backups; 

            pure_backup_value_estimate = pure_backup_value_estimate && child_closest_vertex->pure_backup_value_estimate;
        }

        // Update simplex map
        closest_vertex->value_estimate = avg_val + local_reward;
        closest_vertex->entropy = avg_entropy;
        closest_vertex->pure_backup_value_estimate = pure_backup_value_estimate;

        // simplex map - splitting + message passing
        // should be safe to push, because if value better, the child decision nodes would pick it
        // dont want to pull outdated value estimates though
        simplex->maybe_subdivide(simplex_map, manager);
        closest_vertex->share_values_message_passing_push();
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<SmtThtsDNode> SmtDentsCNode::create_child_node_helper(shared_ptr<const State> next_state) const 
    {
        shared_ptr<SmtBtsDNode> new_child = make_shared<SmtBtsDNode>(
            static_pointer_cast<SmtDentsManager>(thts_manager), 
            next_state, 
            decision_depth+1, 
            decision_timestep+1, 
            static_pointer_cast<const SmtDentsCNode>(shared_from_this()));
        return static_pointer_cast<SmtThtsDNode>(new_child);
    }
}