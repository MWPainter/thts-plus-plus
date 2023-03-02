#include "algorithms/idents_decision_node.h"

#include "algorithms/common/decaying_temp.h"

using namespace std; 

static double EPS = 1e-16;
static double MIN_LOG_WEIGHT = -325.0;

namespace thts {
    IDentsDNode::IDentsDNode(
        shared_ptr<IDentsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const IDentsCNode> parent) :
            MentsDNode(
                thts_manager,
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MentsCNode>(parent)),
            ments_local_entropy(0.0),
            ments_subtree_entropy(0.0),
            local_entropy(0.0),
            subtree_entropy(0.0)
    {
    }

    /**
     * The function used by ments to get the temperature to use for computing weights. In IDents this should be the 
     * 'compute_temp'.
     */
    double IDentsDNode::get_temp() const {
        IDentsManager& manager = (IDentsManager&) *thts_manager;
        return manager.compute_temp;
    }

    /**
     * Returns the value of the decayed temperature for use in policy and entropy backup
    */
    double IDentsDNode::get_decayed_temp() const {
        IDentsManager& manager = (IDentsManager&) *thts_manager;
        return compute_decayed_temp(manager.temp, num_visits, manager.min_temp);
    }

    /**
     * Gets the q_value with decayed to use for a child
     * 
     * This is a copy and paste of get_soft_q_value, but handles accounting for entropy correction
     * 
     * Needs to be a seperate function so that we can keep the original ments backup
     */
    double IDentsDNode::get_decayed_soft_q_value(std::shared_ptr<const Action> action, double opp_coeff) const {
        if (has_child_node(action)) {
            IDentsCNode& child = (IDentsCNode&) *get_child_node(action);
            return opp_coeff * (child.soft_value 
                - get_temp()*child.ments_subtree_entropy 
                + get_decayed_temp()*child.subtree_entropy);
        } 

        if (has_prior()) {
            IDentsManager& manager = (IDentsManager&) *thts_manager;
            double weight = policy_prior->at(action);
            if (weight <= 0.0) {
                // double log_weight = MIN_LOG_WEIGHT; // < log(numeeric_limits<double>::min())
                return MIN_LOG_WEIGHT + prior_shift + manager.prior_policy_boost;
            }
            return (log(weight) + prior_shift + manager.prior_policy_boost);
        } 

        IDentsManager& manager = (IDentsManager&) *thts_manager;
        return manager.default_q_value * opp_coeff;
    }

    /**
     * Compute action weights for this node.
     * 
     * Because we want to sample actions with weights updated according to a decaying temperature, we add that here.
     * 
     * This is a copy and paste of Ments version of this, but with two changes:
     * 1. changing get_soft_q_value for get_decayed_soft_q_value
     * 2. changing get_temp for get_decayed_temp
    */
    void IDentsDNode::compute_action_weights(
        ActionDistr& action_weights, 
        double& sum_action_weights, 
        double& normalisation_term, 
        ThtsEnvContext& context) const
    {
        // get temp
        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        double temp = get_decayed_temp();

        // compute normalisation term
        normalisation_term = numeric_limits<double>::lowest();
        for (shared_ptr<const Action> action : *actions) {
            double q_value_over_temp = get_decayed_soft_q_value(action,opp_coeff) / temp;
            if (normalisation_term < q_value_over_temp) {
                normalisation_term = q_value_over_temp;
            }
        }

        // compute action weights
        sum_action_weights = 0.0;
        for (shared_ptr<const Action> action : *actions) {
            double soft_q_value = get_decayed_soft_q_value(action,opp_coeff);
            double action_weight = exp((soft_q_value/temp) - normalisation_term);
            action_weights[action] = action_weight;
            sum_action_weights += action_weight;
        }
    }

    /**
     * Actually want this distribution for entropy calculations urrgggghhhhhh, code so ugly now
    */
    void IDentsDNode::compute_ments_action_distribution(
        ActionDistr& action_distr, 
        ThtsEnvContext& context) const 
    {  
        // compute boltzmann weights
        double sum_weights;
        double _normalisation_term;
        lock_all_children();
        MentsDNode::compute_action_weights(action_distr, sum_weights, _normalisation_term, context);
        unlock_all_children();

        // compute lambda
        MentsManager& manager = (MentsManager&) *thts_manager;
        double epsilon = manager.epsilon;
        if (is_root_node()) epsilon += manager.root_node_extra_epsilon;
        double lambda = epsilon / log(num_visits+1);
        if (lambda > manager.max_explore_prob) {
            lambda = manager.max_explore_prob;
        }

        // normalise and interpolate masses with uniform masses
        double num_actions = actions->size();
        double uniform_distr_mass = 1.0 / num_actions;
        vector<shared_ptr<const Action>> near_zero_prob_actions;
        for (shared_ptr<const Action> action : *actions) {
            action_distr[action] *= (1.0 - lambda) / sum_weights;
            if (manager.prior_policy_search_weight > 0.0) {
                double lambda_tilde = manager.prior_policy_search_weight;
                if (num_visits >= 2) lambda_tilde /= log(num_visits+1);
                action_distr[action] *= (1.0 - lambda_tilde);
                action_distr[action] += (1.0 - lambda) * lambda_tilde * policy_prior->at(action);
            }
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

    /**
     * This is a copy and past of ments version of this code, but in this case we want to make sure that we still use 
     * the ments version of compute action weights for the backup
    */
    void IDentsDNode::backup_soft(ThtsEnvContext& ctx) {
        ActionDistr action_weights;
        double sum_weights;
        double normalisation_term;
        lock_all_children();
        MentsDNode::compute_action_weights(action_weights, sum_weights, normalisation_term, ctx);
        unlock_all_children();

        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        double temp = get_temp();
        soft_value = opp_coeff * temp * (log(sum_weights) + normalisation_term);

        num_backups++;     
    }

    /**
     * Updates the values of the entropies.
     * 
     * Computes the local entropy of the policy at this node
     * 
     * In the two player case, subtree_entropy = entropy_player - entropy_opponent, assuming we have it computed at 
     * subnodes, then we need to do the following:
     * 1. compute expected value of child entropy given the current local policy
     * 2. add local entropy
     * 
     * N.B. with some maths, we could show H = H_local + sum(Pr(a) * H(a)), where:
     * H = subtree entropy
     * H_local = local entropy
     * Pr(a) = prob select action a
     * H(a) = subtree entropy of child node corresponding to action a
    */
    void IDentsDNode::backup_entropy(ThtsEnvContext& ctx) {
        // Compute local entropy (already thread safe)
        ActionDistr ments_action_distr;
        compute_ments_action_distribution(ments_action_distr, ctx);
        ments_local_entropy = 0.0;
        for (pair<shared_ptr<const Action>,double> pr : ments_action_distr) {
            double prob = pr.second;
            ments_local_entropy -= prob * log(prob);
        }

        // Update subtree entropy == sum child subtree entropies + local
        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        ments_subtree_entropy = opp_coeff * ments_local_entropy;
        lock_all_children();
        for (pair<shared_ptr<const Action>,shared_ptr<ThtsCNode>> pr : children) {
            shared_ptr<const Action> action = pr.first;
            IDentsCNode& child = (IDentsCNode&) *pr.second;
            ments_subtree_entropy += ments_action_distr[action] * child.ments_subtree_entropy;
        }
        unlock_all_children();

        // Compute local entropy (already thread safe)
        ActionDistr action_distr;
        compute_action_distribution(action_distr, ctx);
        local_entropy = 0.0;
        for (pair<shared_ptr<const Action>,double> pr : action_distr) {
            double prob = pr.second;
            local_entropy -= prob * log(prob);
        }

        // Update subtree entropy == sum child subtree entropies + local
        // double opp_coeff = is_opponent() ? -1.0 : 1.0;
        subtree_entropy = opp_coeff * local_entropy;
        lock_all_children();
        for (pair<shared_ptr<const Action>,shared_ptr<ThtsCNode>> pr : children) {
            shared_ptr<const Action> action = pr.first;
            IDentsCNode& child = (IDentsCNode&) *pr.second;
            subtree_entropy += action_distr[action] * child.subtree_entropy;
        }
        unlock_all_children();
    }
            
    /**
     * Just calls the soft and entropy backups
     */
    void IDentsDNode::backup(
        const std::vector<double>& trial_rewards_before_node, 
        const std::vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsEnvContext& ctx) 
    {
        backup_soft(ctx);
        backup_entropy(ctx);   
    }

    /**
     * Return string of the soft value
     */
    string IDentsDNode::get_pretty_print_val() const {
        stringstream ss;
        ss << soft_value 
            << "(t:" << get_decayed_temp() 
            << ",e:" << subtree_entropy 
            << ",dp:" << soft_value - get_temp() * subtree_entropy 
            << ")";
        return ss.str();
    }

    /**
     * Make child node
     */
    shared_ptr<IDentsCNode> IDentsDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<IDentsCNode>(
            static_pointer_cast<IDentsManager>(thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const IDentsDNode>(shared_from_this()));
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    shared_ptr<ThtsCNode> IDentsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<IDentsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}