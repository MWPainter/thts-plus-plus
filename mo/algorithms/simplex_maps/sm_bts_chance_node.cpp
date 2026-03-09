#include "mo/algorithms/simplex_maps/sm_bts_chance_node.h"

#include <iostream>

using namespace std; 

namespace thts {
    SmBtsCNode::SmBtsCNode(
        shared_ptr<SmBtsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const SmBtsDNode> parent) :
            SmThtsCNode(
                static_pointer_cast<SmThtsManager>(thts_manager),
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const SmThtsDNode>(parent)),
            num_backups(0),
            local_reward(Vec::Zero(thts_manager->reward_dim))
    {
        MoThtsEnv& env = *dynamic_pointer_cast<MoThtsEnv>(thts_manager->thts_env());
        local_reward = Vec(env.get_mo_reward_itfc(state,action,*thts_manager->get_thts_context()));
    }

    shared_ptr<const State> SmBtsCNode::sample_observation(MoThtsContext& ctx) 
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
    void SmBtsCNode::backup(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        MoThtsContext& ctx) 
    {  
        SmBtsManager& manager = (SmBtsManager&) *thts_manager;
        num_backups++;

        // Lookup closest vertex and simplex to update
        SMVertexSMSimplexPair vertex_simplex = this->simplex_map.get_closest_vertex_and_adjoining_simplex(
            ctx.context_weight);
        shared_ptr<SMVertex> closest_vertex = vertex_simplex.first;
        shared_ptr<SMSimplex> simplex_to_update = vertex_simplex.second;
        Vec closest_vertex_weight = closest_vertex->weight;

        // Compute backup value as avg of children's
        Vec new_value = Vec::Zero(manager.reward_dim);
        Vec new_value_local = Vec::Zero(manager.reward_dim);

        double sum_child_n_selections = 0;
        for (pair<shared_ptr<const Observation>,shared_ptr<ThtsDNode>> pr : children) {
            shared_ptr<const Observation> observation = pr.first;
            SmBtsDNode& child = (SmBtsDNode&) *pr.second;
            double child_n_selections = empirical_distribution[observation];

            SMVertex& child_vertex = *child.simplex_map.get_closest_vertex(
                closest_vertex_weight, child.simplex_map.get_simplex(closest_vertex_weight));
            Vec child_value = child_vertex.value_estimate;
            Vec child_value_local = child_vertex.value_estimate_local;

            sum_child_n_selections += child_n_selections;

            new_value *= (sum_child_n_selections - child_n_selections) / sum_child_n_selections;
            new_value += child_n_selections * child_value / sum_child_n_selections;

            new_value_local *= (sum_child_n_selections - child_n_selections) / sum_child_n_selections;
            new_value_local += child_n_selections * child_value_local / sum_child_n_selections;
        }

        new_value += local_reward;
        new_value_local += local_reward;

        // Update value in simplex map vertex
        this->simplex_map.update_vertex_values_and_share(
            manager,
            closest_vertex,
            manager.max_push_radius,
            manager.max_neighbours_to_push_to,
            new_value,
            new_value_local,
            0.0 // entropy estimate is not used for BTS
        );

        // And maybe refine the mesh
        this->simplex_map.maybe_subdivide(
            simplex_to_update, 
            manager.min_simplex_radius_in_simplex_tree, 
            manager.max_depth_in_simplex_tree, 
            manager.simplex_split_counter_threshold);

        // Update solved value 
        update_solved_value();
    }

    string SmBtsCNode::get_pretty_print_val() const 
    {
        return "";
    }

    string SmBtsCNode::get_simplex_map_pretty_print_string() const
    {
        return simplex_map.get_pretty_print_string();
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<SmThtsDNode> SmBtsCNode::create_child_node_helper(shared_ptr<const State> next_state) const 
    {
        shared_ptr<SmBtsDNode> new_child = make_shared<SmBtsDNode>(
            static_pointer_cast<SmBtsManager>(thts_manager), 
            next_state, 
            decision_depth+1, 
            decision_timestep+1, 
            static_pointer_cast<const SmBtsCNode>(shared_from_this()));
        return static_pointer_cast<SmThtsDNode>(new_child);
    }

    shared_ptr<SmBtsDNode> SmBtsCNode::create_child_node(shared_ptr<const State> next_state) 
    {
        shared_ptr<const Observation> obs_itfc = static_pointer_cast<const Observation>(next_state);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::create_child_node_itfc(obs_itfc);
        return static_pointer_cast<SmBtsDNode>(new_child);
    }

    shared_ptr<SmBtsDNode> SmBtsCNode::get_child_node(shared_ptr<const State> next_state) const 
    {
        shared_ptr<const Observation> obs_itfc = static_pointer_cast<const Observation>(next_state);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::get_child_node_itfc(obs_itfc);
        return static_pointer_cast<SmBtsDNode>(new_child);
    }
}