#include "mo/algorithms/simplex_maps/sm_dents_chance_node.h"

using namespace std; 

namespace thts {
    SmDentsCNode::SmDentsCNode(
        shared_ptr<SmDentsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const SmDentsDNode> parent) :
            SmBtsCNode(
                static_pointer_cast<SmBtsManager>(thts_manager),
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const SmBtsDNode>(parent))
    {
    }

    /**
     * Todo long term: be less wastful with code copy
    */

    /**
    Backup is same as BTS
    But adds entropy backups (marked with ++DENTS)
     */
    void SmDentsCNode::backup(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        MoThtsContext& ctx) 
    {  
        SmDentsManager& manager = (SmDentsManager&) *thts_manager;
        num_backups++;

        // Lookup closest vertex and simplex to update
        SMVertexSMSimplexPair vertex_simplex = this->simplex_map.get_closest_vertex_and_adjoining_simplex(
            ctx.context_weight);
        shared_ptr<SMVertex> closest_vertex = vertex_simplex.first;
        shared_ptr<SMSimplex> simplex_to_update = vertex_simplex.second;
        Vec closest_vertex_weight = closest_vertex->weight;

        // Compute backup value as avg of children's
        Vec new_value = Vec::Zero(manager.reward_dim);
        Vec new_value_for_search = Vec::Zero(manager.reward_dim);
        double subtree_entropy = 0.0; // ++DENTS

        double sum_child_n_selections = 0;
        for (pair<shared_ptr<const Observation>,shared_ptr<ThtsDNode>> pr : children) {
            shared_ptr<const Observation> observation = pr.first;
            SmBtsDNode& child = (SmBtsDNode&) *pr.second;
            double child_n_selections = empirical_distribution[observation];

            SMVertex& child_vertex = *child.simplex_map.get_closest_vertex(
                closest_vertex_weight, child.simplex_map.get_simplex(closest_vertex_weight));
            Vec child_value = child_vertex.value_estimate;
            Vec child_value_for_search = child_vertex.value_estimate_for_search;
            double child_entropy = child_vertex.entropy_estimate; // ++DENTS

            sum_child_n_selections += child_n_selections;

            new_value *= (sum_child_n_selections - child_n_selections) / sum_child_n_selections;
            new_value += child_n_selections * child_value / sum_child_n_selections;

            new_value_for_search *= (sum_child_n_selections - child_n_selections) / sum_child_n_selections;
            new_value_for_search += child_n_selections * child_value_for_search / sum_child_n_selections;

            subtree_entropy += child_n_selections * child_entropy / sum_child_n_selections; // ++DENTS 
        }

        new_value += local_reward;
        new_value_for_search += local_reward;

        // Update value in simplex map vertex
        this->simplex_map.update_vertex_values_and_share(
            manager,
            closest_vertex,
            manager.max_push_radius,
            manager.max_neighbours_to_push_to,
            new_value,
            new_value_for_search,
            subtree_entropy // ++DENTS
        );

        // And maybe refine the mesh
        this->simplex_map.maybe_subdivide(
            simplex_to_update, 
            manager.min_simplex_radius_in_simplex_tree, 
            manager.max_depth_in_simplex_tree, 
            manager.simplex_split_counter_threshold);
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<SmThtsDNode> SmDentsCNode::create_child_node_helper(shared_ptr<const State> next_state) const 
    {
        shared_ptr<SmBtsDNode> new_child = make_shared<SmBtsDNode>(
            static_pointer_cast<SmDentsManager>(thts_manager), 
            next_state, 
            decision_depth+1, 
            decision_timestep+1, 
            static_pointer_cast<const SmDentsCNode>(shared_from_this()));
        return static_pointer_cast<SmThtsDNode>(new_child);
    }
}