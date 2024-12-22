#include "mo/smt_bts_chance_node.h"

#include <iostream>

using namespace std; 

namespace thts {
    SmtBtsCNode::SmtBtsCNode(
        shared_ptr<SmtBtsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const SmtBtsDNode> parent) :
            SmtThtsCNode(
                static_pointer_cast<SmtThtsManager>(thts_manager),
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const SmtThtsDNode>(parent)),
            num_backups(0),
            local_reward()
    {
        MoThtsEnv& env = *dynamic_pointer_cast<MoThtsEnv>(thts_manager->thts_env());
        local_reward = env.get_mo_reward_itfc(state,action,*thts_manager->get_thts_context());
    }
    
    void SmtBtsCNode::visit(MoThtsContext& ctx) 
    {
        num_visits += 1;
    }  

    shared_ptr<const State> SmtBtsCNode::sample_observation(MoThtsContext& ctx) 
    {
        shared_ptr<const Observation> obs = thts_manager->thts_env()->sample_transition_distribution_itfc(
            state, action, *thts_manager, ctx); 
        shared_ptr<const State> next_state = static_pointer_cast<const State>(obs);
        if (!has_child_node_itfc(obs)) {
            create_child_node(next_state);
        }
        return next_state;
    }


    /**
     * See comments on NGV datatype for what the pure_backup stuff is about
     */
    void SmtBtsCNode::backup(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        MoThtsContext& ctx) 
    {  
        SmtBtsManager& manager = (SmtBtsManager&) *thts_manager;
        num_backups++;
        
        // Get closest NGV in simplex map
        shared_ptr<TN> simplex = simplex_map.get_leaf_tn_node(ctx.context_weight);
        shared_ptr<NGV> closest_vertex = simplex->get_closest_ngv_vertex(ctx.context_weight);

        // Make list of vertices to backup
        vector<shared_ptr<NGV>> vertices_to_backup;
        vertices_to_backup.push_back(closest_vertex);
        if (manager.backup_all_vertices_of_simplex) {
            vector<shared_ptr<NGV>>& simplex_vertices = *simplex->simplex_vertices;
            vertices_to_backup.insert(vertices_to_backup.end(), simplex_vertices.begin(), simplex_vertices.end());
        }

        for (shared_ptr<NGV> ngv : vertices_to_backup) {
            // Compute average value from children
            Eigen::ArrayXd avg_val = Eigen::ArrayXd::Zero(manager.reward_dim);
            bool pure_backup_value_estimate = true;
            int sum_child_backups = 0;
            for (pair<shared_ptr<const Observation>,shared_ptr<ThtsDNode>> pr : children) {
                SmtBtsDNode& child = (SmtBtsDNode&) *pr.second;
                lock_guard<mutex> lg(child.get_lock());
                if (child.num_backups == 0) continue;

                shared_ptr<TN> child_simplex = child.simplex_map.get_leaf_tn_node(ngv->weight);
                shared_ptr<NGV> child_ngv = child_simplex->get_closest_ngv_vertex(ngv->weight);

                sum_child_backups += child.num_backups;
                avg_val *= (sum_child_backups - child.num_backups) / sum_child_backups;
                avg_val += child.num_backups * child_ngv->value_estimate / sum_child_backups; 

                pure_backup_value_estimate = pure_backup_value_estimate && child_ngv->pure_backup_value_estimate;
            }

            // Update simplex map
            ngv->value_estimate = avg_val + local_reward;
            ngv->pure_backup_value_estimate = pure_backup_value_estimate;
        }

        // simplex map - splitting + message passing
        // should be safe to push, because if value better, the child decision nodes would pick it
        // dont want to pull outdated value estimates though
        for (shared_ptr<NGV> ngv : vertices_to_backup) {
            simplex->maybe_subdivide(manager);
            closest_vertex->share_values_message_passing_push();
        }
    }

    string SmtBtsCNode::get_pretty_print_val() const 
    {
        return "";
    }

    string SmtBtsCNode::get_simplex_map_pretty_print_string() const
    {
        return simplex_map.get_pretty_print_string();
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<SmtThtsDNode> SmtBtsCNode::create_child_node_helper(shared_ptr<const State> next_state) const 
    {
        shared_ptr<SmtBtsDNode> new_child = make_shared<SmtBtsDNode>(
            static_pointer_cast<SmtBtsManager>(thts_manager), 
            next_state, 
            decision_depth+1, 
            decision_timestep+1, 
            static_pointer_cast<const SmtBtsCNode>(shared_from_this()));
        return static_pointer_cast<SmtThtsDNode>(new_child);
    }

    shared_ptr<SmtBtsDNode> SmtBtsCNode::create_child_node(shared_ptr<const State> next_state) 
    {
        shared_ptr<const Observation> obs_itfc = static_pointer_cast<const Observation>(next_state);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::create_child_node_itfc(obs_itfc);
        return static_pointer_cast<SmtBtsDNode>(new_child);
    }

    shared_ptr<SmtBtsDNode> SmtBtsCNode::get_child_node(shared_ptr<const State> next_state) const 
    {
        shared_ptr<const Observation> obs_itfc = static_pointer_cast<const Observation>(next_state);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::get_child_node_itfc(obs_itfc);
        return static_pointer_cast<SmtBtsDNode>(new_child);
    }
}