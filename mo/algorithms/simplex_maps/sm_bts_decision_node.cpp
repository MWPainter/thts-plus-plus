#include "mo/algorithms/simplex_maps/sm_bts_decision_node.h"

#include "helper_templates.h"
#include "algorithms/common/decaying_temp.h"
#include "mo/mo_helper.h"

#include <limits>
#include <sstream>

#include <iostream>

using namespace std; 

namespace thts {
    SmBtsDNode::SmBtsDNode(
        shared_ptr<SmBtsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const SmBtsCNode> parent) :
            SmThtsDNode(
                static_pointer_cast<SmThtsManager>(thts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const SmThtsCNode>(parent)),
            num_backups(0)
    {
    }

    double SmBtsDNode::get_temp(MoThtsContext& ctx) const {
        SmBtsManager& manager = (SmBtsManager&) *thts_manager;
        Schedule& temp_schedule = *manager.temp_schedule_ptr;
        return temp_schedule(get_num_visits(ctx));
    }

    /** */
    void SmBtsDNode::read_values_from_child_(
        shared_ptr<const Action> action, 
        Vec& weight,
        int& value_estimate_num_updates_,
        Vec& value_estimate_,
        Vec& value_estimate_for_search_,
        double& entropy_estimate_) const
    {
        // Lookup closest SMVertex in child's simplex map
        SmBtsCNode& child = (SmBtsCNode&) *get_child_node(action);
        shared_ptr<SMVertex> closest_vertex = child.simplex_map.get_closest_vertex(weight);

        // Read values
        value_estimate_num_updates_ = closest_vertex->num_updates;
        value_estimate_ = closest_vertex->value_estimate;
        value_estimate_for_search_ = closest_vertex->value_estimate_for_search;
        entropy_estimate_ = closest_vertex->entropy_estimate;
    }


    /**
    When no child, want to have a default utility
    Can insure this by setting the value estimate to a vector of that default utility
    w^T v will then give the default utility, as ||w||=1
     */
    void SmBtsDNode::fill_child_values_maps_(
        ActionVector& actions,
        Vec& weight,
        unordered_map<shared_ptr<const Action>,int>& value_estimate_num_updates_map_,
        unordered_map<shared_ptr<const Action>,Vec>& value_estimate_map_,
        unordered_map<shared_ptr<const Action>,Vec>& value_estimate_for_search_map_,
        unordered_map<shared_ptr<const Action>,double>& entropy_estimate_map_) const
    {
        SmBtsManager& manager = (SmBtsManager&) *thts_manager;
        int dim = manager.reward_dim;

        for (shared_ptr<const Action> action : actions) 
        {
            // Fill with default values (makes sure something exists in maps)
            value_estimate_num_updates_map_[action] = 0;
            value_estimate_map_.emplace(action, Vec(dim, manager.default_q_utility));
            value_estimate_for_search_map_.emplace(action, Vec(dim, manager.default_q_utility));
            entropy_estimate_map_[action] = 0.0;

            // Overwrite with child values if child exists
            if (has_child_node_itfc(action)) 
            {
                this->read_values_from_child_(
                    action,
                    weight,
                    value_estimate_num_updates_map_[action],
                    value_estimate_map_.at(action),
                    value_estimate_for_search_map_.at(action),
                    entropy_estimate_map_[action]
                );
            }
        }
    }

    unordered_map<shared_ptr<const Action>,double> SmBtsDNode::utility_weights_from_values(
        Vec& weight,
        unordered_map<shared_ptr<const Action>,Vec>& values) const
    {
        unordered_map<shared_ptr<const Action>,double> utility_weights;
        for (pair<shared_ptr<const Action>,Vec> pr : values) 
        {
            shared_ptr<const Action> action = pr.first;
            Vec value = pr.second;
            utility_weights[action] = weight.dot(value);
        }
        return utility_weights;
    }

    /**
    Compute action weights
    Read values in 
    Pass to helper
    */
    void SmBtsDNode::compute_action_weights_(
        ActionVector& actions,  
        MoThtsContext& context,
        ActionDistr& action_weights_,
        double& sum_weights_) const
    {
        // read values
        unordered_map<shared_ptr<const Action>,int> _value_estimate_num_updates_map;
        unordered_map<shared_ptr<const Action>,Vec> _value_estimate_map;
        unordered_map<shared_ptr<const Action>,Vec> value_estimate_for_search_map;
        unordered_map<shared_ptr<const Action>,double> _entropy_estimate_map;
        this->fill_child_values_maps_(
            actions, 
            context.context_weight, 
            _value_estimate_num_updates_map, 
            _value_estimate_map, 
            value_estimate_for_search_map, 
            _entropy_estimate_map);

        // call helper
        this->compute_action_weights_helper_(
            actions,
            context,
            value_estimate_for_search_map,
            _entropy_estimate_map,
            action_weights_,
            sum_weights_);
    }
    
    /**
    Actually computes boltzmann action weights, given value estimates from children
    Updates action_weights_ and sum_weights_
    */
    void SmBtsDNode::compute_action_weights_helper_(
        ActionVector& actions,
        MoThtsContext& context,
        unordered_map<shared_ptr<const Action>,Vec>& value_estimate_for_search_map,
        unordered_map<shared_ptr<const Action>,double>& entropy_estimate_map,
        ActionDistr& action_weights_,
        double& sum_weights_) const
    {
        SmBtsManager& manager = (SmBtsManager&) *thts_manager;

        // get temp
        double temp = this->get_temp(context);

        // compute utility weights
        unordered_map<shared_ptr<const Action>,double> q_utilities = this->utility_weights_from_values(
            context.context_weight, value_estimate_for_search_map);

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
    Compute full action distribution used by BTS
    Passes to helpers
    First gets boltzmann weights
    Second adds the epsilon greedy mass
    */
    void SmBtsDNode::compute_action_distribution_(
        ActionVector& actions,
        MoThtsContext& context,
        ActionDistr& action_distr_) const
    {  
        // start with boltzmann weights
        double sum_weights = 0.0;
        this->compute_action_weights_(actions, context, action_distr_, sum_weights);
        this->compute_action_distribution_helper_(actions, context, action_distr_, sum_weights);
    }

    /**
    Adds the epsilon greedy mass to the boltzmann distribution
    Updates action_distr_
    */
    void SmBtsDNode::compute_action_distribution_helper_(
        ActionVector& actions,
        MoThtsContext& context,
        ActionDistr& action_distr_,
        double sum_weights) const
    {
        // compute lambda
        SmBtsManager& manager = (SmBtsManager&) *thts_manager;
        double epsilon = manager.epsilon;
        double lambda = epsilon / log(get_num_visits(context)+1);
        if (lambda > manager.max_explore_prob) {
            lambda = manager.max_explore_prob;
        }

        // normalise and interpolate masses with uniform masses
        double num_actions = actions.size();
        double uniform_distr_mass = 1.0 / num_actions;
        for (shared_ptr<const Action> action : actions) {
            action_distr_[action] *= (1.0 - lambda) / sum_weights;
            // if (manager.prior_policy_search_weight > 0.0) {
            //     double lambda_tilde = manager.prior_policy_search_weight / log(get_num_visits(ctx)+3);
            //     action_distr[action] *= (1.0 - lambda_tilde);
            //     action_distr[action] += (1.0 - lambda) * lambda_tilde * policy_prior->at(action);
            // }
            action_distr_[action] += lambda * uniform_distr_mass;
        }
    }

    shared_ptr<const Action> SmBtsDNode::select_action(MoThtsContext& ctx) 
    {
        // Compute action distribution
        shared_ptr<ActionVector> actions = thts_manager->thts_env()->get_valid_actions_itfc(state, ctx);
        ActionDistr action_distr_;
        this->compute_action_distribution_(*actions, ctx, action_distr_);

        // Sample, create child node if needed, and return
        shared_ptr<const Action> selected_action = helper::sample_from_distribution(action_distr_, *thts_manager);
        if (!has_child_node_itfc(selected_action)) {
            create_child_node(selected_action);
        }
        return selected_action;
    }

    shared_ptr<const Action> SmBtsDNode::recommend_action(MoThtsContext& ctx) const 
    { 
        // Read values from children
        shared_ptr<ActionVector> actions = thts_manager->thts_env()->get_valid_actions_itfc(state, ctx);
        unordered_map<shared_ptr<const Action>,int> _value_estimate_num_updates_map;
        unordered_map<shared_ptr<const Action>,Vec> q_vals_;
        unordered_map<shared_ptr<const Action>,Vec> _q_vals_for_search_;
        unordered_map<shared_ptr<const Action>,double> _entropy_map_;
        this->fill_child_values_maps_(
            *actions, 
            ctx.context_weight, 
            _value_estimate_num_updates_map, 
            q_vals_, 
            _q_vals_for_search_, 
            _entropy_map_);

        // If no values added, return a random action
        if (q_vals_.size() == 0) {
            int indx = thts_manager->get_rand_int(0, static_cast<int>(actions->size()));
            return actions->at(indx);
        }

        // Compute utility weights
        unordered_map<shared_ptr<const Action>,double> q_utilities = this->utility_weights_from_values(ctx.context_weight, q_vals_);

        // Return action with max utility
        return helper::get_max_key_break_ties_randomly(q_utilities, *thts_manager);
    }

    /**
     * See comments on NGV datatype for what the pure_backup stuff is about
     */
    void SmBtsDNode::backup(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        MoThtsContext& ctx) 
    {
        SmBtsManager& manager = (SmBtsManager&) *thts_manager;
        
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
        unordered_map<shared_ptr<const Action>,double> _entropy_map;
        this->fill_child_values_maps_( 
            *actions, 
            closest_vertex_weight, 
            q_vals_num_updates, 
            q_vals, 
            q_vals_for_search, 
            _entropy_map);

        // Compute a new value
        double new_utility = std::numeric_limits<double>::lowest();
        Vec new_value = Vec(manager.reward_dim, 0.0);
        double new_utility_for_search = std::numeric_limits<double>::lowest();
        Vec new_value_for_search = Vec(manager.reward_dim, 0.0);

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
        }

        // If have a heuristic value, mix it in
        if (this->has_heuristic_value())
        {
            SmBtsManager& manager = (SmBtsManager&) *thts_manager;
            new_value_for_search *= (num_backups - manager.heuristic_weight) / num_backups;
            new_value_for_search += manager.heuristic_weight * heuristic_value / num_backups;
        }
        
        // Actually update the SMVertex with the new values
        this->simplex_map.update_vertex_values_and_share(
            manager,
            closest_vertex,
            manager.max_push_radius,
            manager.max_neighbours_to_push_to,
            new_value,
            new_value_for_search,
            0.0 // entropy estimate is not used for BTS
        );

        // And maybe refine the mesh
        this->simplex_map.maybe_subdivide(
            simplex_to_update, 
            manager.min_simplex_radius_in_simplex_tree, 
            manager.max_depth_in_simplex_tree, 
            manager.simplex_split_counter_threshold);
    }

    string SmBtsDNode::get_pretty_print_val() const {
        return "";
    }

    string SmBtsDNode::get_simplex_map_pretty_print_string() const
    {
        return simplex_map.get_pretty_print_string();
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<SmBtsCNode> SmBtsDNode::create_child_node(shared_ptr<const Action> action) 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::create_child_node_itfc(act_itfc);
        return static_pointer_cast<SmBtsCNode>(new_child);
    }

    shared_ptr<SmThtsCNode> SmBtsDNode::create_child_node_helper(shared_ptr<const Action> action) const 
    {   
        shared_ptr<SmBtsCNode> new_child = make_shared<SmBtsCNode>(
            static_pointer_cast<SmBtsManager>(thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const SmBtsDNode>(shared_from_this()));
        return static_pointer_cast<SmThtsCNode>(new_child);
    }

    shared_ptr<SmBtsCNode> SmBtsDNode::get_child_node(shared_ptr<const Action> action) const
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::get_child_node_itfc(act_itfc);
        return static_pointer_cast<SmBtsCNode>(new_child);
    }
}