#include "mo/algorithms/simplex_maps/sm_dents_decision_node.h"

#include "helper_templates.h"
#include "algorithms/common/decaying_temp.h"
#include "mo/mo_helper.h"

#include <limits>
#include <sstream>

using namespace std; 

namespace thts {
    SmDentsDNode::SmDentsDNode(
        shared_ptr<SmDentsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const SmDentsCNode> parent) :
            SmBtsDNode(
                static_pointer_cast<SmBtsManager>(thts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const SmBtsCNode>(parent))
    {
    }

    double SmDentsDNode::get_entropy_coeff(MoThtsContext& ctx) const {
        SmDentsManager& manager = (SmDentsManager&) *thts_manager;
        Schedule& entropy_coeff_schedule = *manager.entropy_coeff_schedule_ptr;
        return entropy_coeff_schedule(get_num_visits(ctx));
    }
    
    /**
    Actually computes boltzmann action weights, given value estimates from children
    Updates action_weights_ and sum_weights_
    */
    void SmDentsDNode::compute_action_weights_helper_(
        ActionVector& actions,
        MoThtsContext& context,
        unordered_map<shared_ptr<const Action>,Vec>& value_estimate_for_search_map,
        unordered_map<shared_ptr<const Action>,double>& entropy_estimate_map,
        ActionDistr& action_weights_,
        double& sum_weights_) const
    {
        SmDentsManager& manager = (SmDentsManager&) *thts_manager;

        // get temp
        double temp = this->get_temp(context);
        double entropy_coeff = this->get_entropy_coeff(context); // ++DENTS

        // compute utility weights
        unordered_map<shared_ptr<const Action>,double> q_utilities = this->utility_weights_from_values(
            context.context_weight, value_estimate_for_search_map);

        // Normalise entropies + Q values before combining (++DENTS)
        if (manager.normalise_entropy_before_adding) {
            thts::helper::linearly_normalise_values(q_utilities);
            thts::helper::linearly_normalise_values(entropy_estimate_map);
        }

        // Update q_utilities with the entropy terms (++DENTS)
        for (shared_ptr<const Action> action : actions) 
        {
            q_utilities[action] += entropy_coeff * entropy_estimate_map[action];
        }

        // Normalise the utilities
        if (manager.normalise_q_values) {
            thts::helper::linearly_normalise_values(q_utilities);
        }

        // Compute normalization term (for the boltzmann distribution, so max weight = exp(0) = 1)
        double normalisation_term = numeric_limits<double>::lowest();
        for (shared_ptr<const Action> action : actions) {
            double utility_over_temp = q_utilities[action] / temp;
            if (normalisation_term < utility_over_temp) {
                normalisation_term = utility_over_temp;
            }
        }

        // compute action weights (boltzmann distribution)
        sum_weights_ = 0.0;
        for (shared_ptr<const Action> action : actions) {
            double action_weight = exp((q_utilities[action]/temp) - normalisation_term);
            action_weights_[action] = action_weight;
            sum_weights_ += action_weight;
        }
    }

    /**
    Backup is same as BTS
    But adds entropy backups (marked with ++DENTS)
     */
    void SmDentsDNode::backup(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        MoThtsContext& ctx) 
    {
        SmDentsManager& manager = (SmDentsManager&) *thts_manager;
        
        // Increment backups
        num_backups++;

        // Lookup closest vertex and simplex to update
        SMVertexSMSimplexPair vertex_simplex = this->simplex_map.get_closest_vertex_and_adjoining_simplex(
            ctx.context_weight);
        shared_ptr<SMVertex> closest_vertex = vertex_simplex.first;
        shared_ptr<SMSimplex> simplex_to_update = vertex_simplex.second;
        Vec closest_vertex_weight = closest_vertex->weight;

        // Get values from children for the weight we're updating
        shared_ptr<ActionVector> actions = thts_manager->thts_env()->get_valid_actions_itfc(state, ctx);
        unordered_map<shared_ptr<const Action>,int> q_vals_num_updates;
        unordered_map<shared_ptr<const Action>,Vec> q_vals;
        unordered_map<shared_ptr<const Action>,Vec> q_vals_for_search;
        unordered_map<shared_ptr<const Action>,double> entropy_map;
        this->fill_child_values_maps_( 
            *actions, 
            closest_vertex_weight, 
            q_vals_num_updates, 
            q_vals, 
            q_vals_for_search, 
            entropy_map);

        // ++DENTS - compute action policy for entropy backups
        ActionDistr policy;
        double sum_weights;
        this->compute_action_weights_helper_(
            *actions,
            ctx,
            q_vals_for_search,
            entropy_map,
            policy,
            sum_weights);
        this->compute_action_distribution_helper_(
            *actions,
            ctx,
            policy,
            sum_weights);

        // Compute a new value
        double new_utility = std::numeric_limits<double>::lowest();
        Vec new_value = Vec::Zero(manager.reward_dim);
        double new_utility_for_search = std::numeric_limits<double>::lowest();
        Vec new_value_for_search = Vec::Zero(manager.reward_dim);
        double subtree_entropy = 0.0; // ++DENTS

        for (shared_ptr<const Action> action : *actions) {
            // If SMVertex at child is not updated, OR, child doesn't exist, skip it (dont allow backups to take default_q_utility)
            if (q_vals_num_updates[action] == 0) {
                continue;
            }
            double q_utility = closest_vertex_weight.dot(q_vals.at(action));
            if (q_utility > new_utility) {
                new_utility = q_utility;
                new_value = q_vals.at(action);
            }
            double q_utility_for_search = closest_vertex_weight.dot(q_vals_for_search.at(action));
            if (q_utility_for_search > new_utility_for_search) {
                new_utility_for_search = q_utility_for_search;
                new_value_for_search = q_vals_for_search.at(action);
            }
            subtree_entropy += policy[action] * entropy_map[action]; // ++DENTS
        }

        // If have a heuristic value, mix it in
        if (this->has_heuristic_value())
        {
            SmBtsManager& manager = (SmBtsManager&) *thts_manager;
            new_value_for_search *= (num_backups - manager.heuristic_weight) / num_backups;
            new_value_for_search += manager.heuristic_weight * heuristic_value / num_backups;
        }

        // ++DENTS - Compute new entropy estimate, with local entropy + subtree entropy
        double local_entropy = 0.0; 
        for (pair<shared_ptr<const Action>,double> pr : policy) {
            double prob = pr.second;
            if (prob == 0.0) continue;
            local_entropy -= prob * log(prob); 
        }
        double new_entropy = local_entropy + subtree_entropy; 
        
        // Actually update the SMVertex with the new values
        this->simplex_map.update_vertex_values_and_share(
            manager,
            closest_vertex,
            manager.max_push_radius,
            manager.max_neighbours_to_push_to,
            new_value,
            new_value_for_search,
            new_entropy // ++DENTS
        );

        // And maybe refine the mesh
        this->simplex_map.maybe_subdivide(
            simplex_to_update, 
            manager.min_simplex_radius_in_simplex_tree, 
            manager.max_depth_in_simplex_tree, 
            manager.simplex_split_counter_threshold);

        // Update solved value and backup stats
        update_solved_value();
        increment_and_update_backup_count();
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<SmThtsCNode> SmDentsDNode::create_child_node_helper(shared_ptr<const Action> action) const 
    {   
        shared_ptr<SmDentsCNode> new_child = make_shared<SmDentsCNode>(
            static_pointer_cast<SmDentsManager>(thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const SmDentsDNode>(shared_from_this()));
        return static_pointer_cast<SmThtsCNode>(new_child);
    }
}