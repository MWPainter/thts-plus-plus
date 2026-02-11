#include "mo/algorithms/chmcts/ch_bts_decision_node.h"

#include "helper_templates.h"

using namespace std; 

// Epsilon to be used as a minimum prob, if lower than this just set to zero
static double EPS = 1e-16;

namespace thts {
    ChBtsDNode::ChBtsDNode(
        shared_ptr<ChBtsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ChBtsCNode> parent) :
            ChThtsDNode(
                static_pointer_cast<ChThtsManager>(thts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const ChThtsCNode>(parent))
    {
    }

    /**
     * Copies from ments
     */
    double ChBtsDNode::get_temp(MoThtsContext& context) const
    {
        ChBtsManager& manager = (ChBtsManager&) *thts_manager;
        Schedule& temp_schedule = *manager.temp_schedule_ptr;
        return temp_schedule(get_num_visits(context));

    }

    void ChBtsDNode::compute_action_weights(
        ActionDistr& action_weights, 
        double& sum_action_weights, 
        double& numerical_stability_term, 
        MoThtsContext& context) const
    {
        // Get q values and temp
        ChBtsManager& manager = (ChBtsManager&) *thts_manager;
        ActionDistr q_values;
        fill_contextual_q_values(q_values, context, manager.default_q_value);
        double temp = get_temp(context);

        // Optionally normalise Q values
        if (manager.normalise_q_values) {
            double min_q_value = numeric_limits<double>::max();
            double max_q_value = numeric_limits<double>::lowest();

            for (pair<shared_ptr<const Action>,double> pr : q_values) {
                double q_value = pr.second;
                if (q_value < min_q_value) min_q_value = q_value;
                if (q_value > max_q_value) max_q_value = q_value;
            }

            for (pair<shared_ptr<const Action>,double> pr : q_values) {
                shared_ptr<const Action> action = pr.first;
                double q_value = pr.second;
                q_values[action] = (q_value - min_q_value) / (max_q_value - min_q_value + EPS);
            }
        }

        // compute (numerical stability) normalisation term
        numerical_stability_term = numeric_limits<double>::lowest();
        for (pair<shared_ptr<const Action>,double> pair : q_values) {
            double q_value = pair.second;
            double q_value_over_temp =  q_value / temp;
            if (q_value_over_temp > numerical_stability_term) {
                numerical_stability_term = q_value_over_temp;
            }
        }

        // compute action weights
        sum_action_weights = 0.0;
        for (pair<shared_ptr<const Action>,double> pr : q_values) {
            shared_ptr<const Action> action = pr.first;
            double q_value = pr.second;
            double action_weight = exp((q_value/temp) - numerical_stability_term);
            action_weights[action] = action_weight;
            sum_action_weights += action_weight;
        }
    }

    void ChBtsDNode::compute_action_distribution(
        ActionDistr& action_distr, 
        MoThtsContext& context) const
    {
        // compute boltzmann weights
        double sum_weights;
        double _normalisation_term;
        compute_action_weights(action_distr, sum_weights, _normalisation_term, context);

        // Avoid division by zero
        if (sum_weights == 0.0) {
            sum_weights = 1.0;
        }

        // compute lambda
        ChBtsManager& manager = (ChBtsManager&) *thts_manager;
        double epsilon = manager.epsilon;
        double lambda = epsilon / log(num_visits+1);
        if (lambda > manager.max_explore_prob) {
            lambda = manager.max_explore_prob;
        }

        // normalise and interpolate masses with uniform masses
        double num_actions = action_distr.size();
        double uniform_distr_mass = 1.0 / num_actions;
        vector<shared_ptr<const Action>> near_zero_prob_actions;
        for (pair<shared_ptr<const Action>,double> pair : action_distr) {
            shared_ptr<const Action> action = pair.first;
            action_distr[action] *= (1.0 - lambda) / sum_weights;
            action_distr[action] += lambda * uniform_distr_mass;
            if (action_distr[action] < EPS) {
                near_zero_prob_actions.push_back(action);
            }
        }

        // Remove close to zero probabilities (never going to sample + leads to numerical ick later)
        for (shared_ptr<const Action> action : near_zero_prob_actions) {
            action_distr.erase(action);
        }
    }

    shared_ptr<const Action> ChBtsDNode::select_action(MoThtsContext& ctx)
    {
        ActionDistr action_distr;
        compute_action_distribution(action_distr, ctx);
        shared_ptr<const Action> selected_action = helper::sample_from_distribution(action_distr, *thts_manager);
        if (!has_child_node_itfc(selected_action)) {
            create_child_node(selected_action);
        }
        return selected_action;
    }

    string ChBtsDNode::get_pretty_print_val() const 
    {
        return "";
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<ChBtsCNode> ChBtsDNode::create_child_node(shared_ptr<const Action> action) 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::create_child_node_itfc(act_itfc);
        return static_pointer_cast<ChBtsCNode>(new_child);
    } 

    /**
     * Added making the child's czt_node pointing to the same CztCNode as our czt_node
    */
    shared_ptr<ChThtsCNode> ChBtsDNode::create_child_node_helper(shared_ptr<const Action> action) const 
    {   
        shared_ptr<ChBtsCNode> child_node = make_shared<ChBtsCNode>(
            static_pointer_cast<ChBtsManager>(thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const ChBtsDNode>(shared_from_this()));
        return static_pointer_cast<ChThtsCNode>(child_node);
    }

    shared_ptr<ChBtsCNode> ChBtsDNode::get_child_node(shared_ptr<const Action> action) const 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::get_child_node_itfc(act_itfc);
        return static_pointer_cast<ChBtsCNode>(new_child);
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    void ChBtsDNode::visit_itfc(ThtsContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        visit(ctx_itfc);
    }

    shared_ptr<const Action> ChBtsDNode::select_action_itfc(ThtsContext& ctx) 
    {
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        return select_action(mo_ctx);
    }

    shared_ptr<const Action> ChBtsDNode::recommend_action_itfc(ThtsContext& ctx) const 
    {
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        return recommend_action(mo_ctx);
    }

    void ChBtsDNode::backup_itfc(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        ThtsContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        backup(
            trial_rewards_before_node, 
            trial_rewards_after_node, 
            trial_cumulative_return_after_node, 
            trial_cumulative_return, 
            ctx_itfc);
    }

    shared_ptr<ThtsCNode> ChBtsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ChThtsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}