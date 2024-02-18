#include "mo/smt_bts_decision_node.h"

#include "helper_templates.h"
#include "algorithms/common/decaying_temp.h"
#include "mo/mo_helper.h"

#include <limits>
#include <sstream>

using namespace std; 

namespace thts {
    SmtBtsDNode::SmtBtsDNode(
        shared_ptr<SmtBtsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const SmtBtsCNode> parent) :
            SmtThtsDNode(
                static_pointer_cast<SmtThtsManager>(thts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const SmtThtsCNode>(parent)),
            num_backups(0)
    {
    }

    double SmtBtsDNode::get_temp() const {
        SmtBtsManager& manager = (SmtBtsManager&) *thts_manager;
        if (manager.temp_decay_fn == nullptr) {
            return manager.temp;
        }
        return compute_decayed_temp(
            manager.temp_decay_fn, 
            manager.temp, 
            manager.temp_decay_min_temp, 
            num_visits, 
            manager.temp_decay_visits_scale);
    }
    
    Eigen::ArrayXd SmtBtsDNode::get_q_value(
        shared_ptr<const Action> action, 
        double opp_coeff, 
        MoThtsContext& ctx,
        double& entropy) const 
    {
        SmtBtsManager& manager = (SmtBtsManager&) *thts_manager;
        if (!has_child_node_itfc(action)) {
            return manager.default_q_value * opp_coeff;
        }

        SmtBtsCNode& child = (SmtBtsCNode&) *get_child_node(action);
        lock_guard<mutex> lg(child.get_lock());
        shared_ptr<TN> simplex = child.simplex_map.get_leaf_tn_node(ctx.context_weight);
        shared_ptr<NGV> closest_vertex = simplex->get_closest_ngv_vertex(ctx.context_weight);
        entropy = closest_vertex->entropy;
        return closest_vertex->value_estimate * opp_coeff;
    }

    void SmtBtsDNode::get_child_q_values(
        ActionVector& actions,
        unordered_map<shared_ptr<const Action>,Eigen::ArrayXd>& q_val_map, 
        unordered_map<shared_ptr<const Action>,double>& entropy_map, 
        MoThtsContext& ctx) const
    {
        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        for (shared_ptr<const Action> action : actions) {
            double entropy;
            q_val_map[action] = get_q_value(action, opp_coeff, ctx, entropy);
            entropy_map[action] = entropy;
        }
    }

    void SmtBtsDNode::compute_action_weights(
        ActionVector& actions,
        unordered_map<shared_ptr<const Action>,Eigen::ArrayXd>& q_val_map,
        unordered_map<shared_ptr<const Action>,double>& entropy_map, 
        ActionDistr& action_weights, 
        double& sum_action_weights, 
        double& normalisation_term, 
        MoThtsContext& context) const
    {
        // get temp
        double temp = get_temp();

        // compute normalisation term
        normalisation_term = numeric_limits<double>::lowest();
        for (shared_ptr<const Action> action : actions) {
            double ctx_val_over_temp = thts::helper::dot(context.context_weight, q_val_map[action]) / temp;
            if (normalisation_term < ctx_val_over_temp) {
                normalisation_term = ctx_val_over_temp;
            }
        }

        // compute action weights
        sum_action_weights = 0.0;
        for (shared_ptr<const Action> action : actions) {
            double ctx_q_value = thts::helper::dot(context.context_weight, q_val_map[action]);
            double action_weight = exp((ctx_q_value/temp) - normalisation_term);
            action_weights[action] = action_weight;
            sum_action_weights += action_weight;
        }
    }

    void SmtBtsDNode::compute_action_distribution(
        ActionVector& actions,
        unordered_map<shared_ptr<const Action>,Eigen::ArrayXd>& q_val_map,
        unordered_map<shared_ptr<const Action>,double>& entropy_map, 
        ActionDistr& action_distr, 
        MoThtsContext& context) const 
    {  
        // compute boltzmann weights
        double sum_weights;
        double _normalisation_term;
        compute_action_weights(
            actions, q_val_map, entropy_map, action_distr, sum_weights, _normalisation_term, context);

        // compute lambda
        SmtBtsManager& manager = (SmtBtsManager&) *thts_manager;
        double epsilon = manager.epsilon;
        if (is_root_node() && manager.root_node_epsilon > 0.0) epsilon = manager.root_node_epsilon;
        double lambda = epsilon / log(num_visits+1);
        if (lambda > manager.max_explore_prob) {
            lambda = manager.max_explore_prob;
        }

        // normalise and interpolate masses with uniform masses
        double num_actions = actions.size();
        double uniform_distr_mass = 1.0 / num_actions;
        for (shared_ptr<const Action> action : actions) {
            action_distr[action] *= (1.0 - lambda) / sum_weights;
            // if (manager.prior_policy_search_weight > 0.0) {
            //     double lambda_tilde = manager.prior_policy_search_weight / log(num_visits+3);
            //     action_distr[action] *= (1.0 - lambda_tilde);
            //     action_distr[action] += (1.0 - lambda) * lambda_tilde * policy_prior->at(action);
            // }
            action_distr[action] += lambda * uniform_distr_mass;
        }
    }
    
    void SmtBtsDNode::visit(MoThtsContext& ctx) 
    {
        num_visits += 1;

        if (is_root_node()) {
            NGV& random_ngv = *simplex_map.sample_random_ngv_vertex(*thts_manager);
            ctx.context_weight = random_ngv.weight;
        }
    } 

    shared_ptr<const Action> SmtBtsDNode::select_action(MoThtsContext& ctx) 
    {
        shared_ptr<ActionVector> actions = thts_manager->thts_env()->get_valid_actions_itfc(state, ctx);
        unordered_map<shared_ptr<const Action>,Eigen::ArrayXd> q_val_map;
        unordered_map<shared_ptr<const Action>,double> entropy_map;
        get_child_q_values(*actions, q_val_map, entropy_map, ctx);

        ActionDistr action_distr;
        compute_action_distribution(*actions, q_val_map, entropy_map, action_distr, ctx);

        shared_ptr<const Action> selected_action = helper::sample_from_distribution(action_distr, *thts_manager);
        if (!has_child_node_itfc(selected_action)) {
            create_child_node(selected_action);
        }
        return selected_action;
    }

    shared_ptr<const Action> SmtBtsDNode::recommend_action(MoThtsContext& ctx) const 
    { 
        shared_ptr<ActionVector> actions = thts_manager->thts_env()->get_valid_actions_itfc(state, ctx);
        unordered_map<shared_ptr<const Action>,Eigen::ArrayXd> q_val_map;
        unordered_map<shared_ptr<const Action>,double> entropy_map;
        get_child_q_values(*actions, q_val_map, entropy_map, ctx);

        unordered_map<shared_ptr<const Action>,double> ctx_q_val_map;
        for (shared_ptr<const Action> action : *actions) {
            ctx_q_val_map[action] = thts::helper::dot(ctx.context_weight, q_val_map[action]);
        }

        return helper::get_max_key_break_ties_randomly(ctx_q_val_map, *thts_manager);
    }

    void SmtBtsDNode::backup(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        MoThtsContext& ctx) 
    {
        num_backups++;

        shared_ptr<ActionVector> actions = thts_manager->thts_env()->get_valid_actions_itfc(state, ctx);
        unordered_map<shared_ptr<const Action>,Eigen::ArrayXd> q_val_map;
        unordered_map<shared_ptr<const Action>,double> entropy_map;
        get_child_q_values(*actions, q_val_map, entropy_map, ctx);

        Eigen::ArrayXd best_q_val;
        double max_ctx_q_val = numeric_limits<double>::lowest();
        for (shared_ptr<const Action> action : *actions) {
            double ctx_q_val = thts::helper::dot(ctx.context_weight, q_val_map[action]);
            if (ctx_q_val > max_ctx_q_val) {
                max_ctx_q_val = ctx_q_val;
                best_q_val = q_val_map[action];
            }
        }

        shared_ptr<TN> simplex = simplex_map.get_leaf_tn_node(ctx.context_weight);
        shared_ptr<NGV> closest_vertex = simplex->get_closest_ngv_vertex(ctx.context_weight);

        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        closest_vertex->value_estimate = best_q_val * opp_coeff;

        SmtBtsManager& manager = (SmtBtsManager&) *thts_manager;
        simplex->maybe_subdivide(simplex_map, manager);
        closest_vertex->share_values_message_passing();
    }

    string SmtBtsDNode::get_pretty_print_val() const {
        return "";
    }

    string SmtBtsDNode::get_simplex_map_pretty_print_string() const
    {
        return simplex_map.get_pretty_print_string();
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<SmtBtsCNode> SmtBtsDNode::create_child_node(shared_ptr<const Action> action) 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::create_child_node_itfc(act_itfc);
        return static_pointer_cast<SmtBtsCNode>(new_child);
    }

    shared_ptr<SmtThtsCNode> SmtBtsDNode::create_child_node_helper(shared_ptr<const Action> action) const 
    {   
        shared_ptr<SmtBtsCNode> new_child = make_shared<SmtBtsCNode>(
            static_pointer_cast<SmtBtsManager>(thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const SmtBtsDNode>(shared_from_this()));
        return static_pointer_cast<SmtThtsCNode>(new_child);
    }

    shared_ptr<SmtBtsCNode> SmtBtsDNode::get_child_node(shared_ptr<const Action> action) const
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::get_child_node_itfc(act_itfc);
        return static_pointer_cast<SmtBtsCNode>(new_child);
    }
}