#include "algorithms/uct_chance_node.h"

#include "helper_templates.h"

using namespace std; 

namespace thts {
    /**
     * Construct Uct Chance node. Use thts_manager to initialise values with heuristic if necessary.
     */
    UctCNode::UctCNode(
        shared_ptr<UctManager> thts_manager,
        shared_ptr<ThtsEnv> thts_env,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const UctDNode> parent) :
            ThtsCNode(
                static_pointer_cast<ThtsManager>(thts_manager),
                thts_env,
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const ThtsDNode>(parent)),
            next_state_distr(thts_env->get_transition_distribution_itfc(state,action)),
            avg_return(0.0)
    {  
        if (thts_manager->use_heuristic_at_chance_nodes && thts_manager->heuristic_fn != nullptr) {
            num_visits = thts_manager->heuristic_psuedo_trials;
            avg_return = thts_manager->heuristic_fn(state, action);
        }
    }

    /**
     * Visit just needs to increment num_visits.
     */
    void UctCNode::visit(ThtsEnvContext& ctx) {
        num_visits += 1;
    }

    /**
     * Implementation of sample_observation, that uses the sample from distribution helper function.
     * 
     * thts_env->sample_observation
     */
    shared_ptr<const State> UctCNode::sample_observation_random() {
        shared_ptr<const State> sampled_state = helper::sample_from_distribution(*next_state_distr, *thts_manager);
        if (!has_child_node(sampled_state)) {
            create_child_node(sampled_state);
        }
        return sampled_state;
    }
    
    /**
     * Sample observation calls sample_observation_random.
     */
    shared_ptr<const State> UctCNode::sample_observation(ThtsEnvContext& ctx) {
        return sample_observation_random();
    }

    /**
     * Computes running average.
     */
    void UctCNode::backup_average_return(const double trial_return_after_node) {
        avg_return += (trial_return_after_node - avg_return) / (double) num_visits;
    }

    /**
     * Calls the running average return backup function.
     */
    void UctCNode::backup(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsEnvContext& ctx) 
    {
        backup_average_return(trial_cumulative_return_after_node);
    }

    /**
     * Make a new UctDNode on the heap, with correct arguments for a child node.
     */
    shared_ptr<UctDNode> UctCNode::create_child_node_helper(shared_ptr<const State> observation) const
    {
        shared_ptr<const State> next_state = static_pointer_cast<const State>(observation);
        return make_shared<UctDNode>(
            static_pointer_cast<UctManager>(thts_manager), 
            thts_env, 
            next_state,
            decision_depth+1, 
            decision_timestep+1, 
            static_pointer_cast<const UctCNode>(shared_from_this()));
    }

    /**
     * Pretty print val = print current avg_return in node
     */
    string UctCNode::get_pretty_print_val() const {
        stringstream ss;
        ss << avg_return;
        return ss.str();
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<UctDNode> UctCNode::create_child_node(shared_ptr<const State> observation) 
    {
        shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::create_child_node_itfc(obsv_itfc);
        return static_pointer_cast<UctDNode>(new_child);
    }

    bool UctCNode::has_child_node(std::shared_ptr<const State> observation) const {
        return ThtsCNode::has_child_node_itfc(static_pointer_cast<const Observation>(observation));
    }
    
    shared_ptr<UctDNode> UctCNode::get_child_node(shared_ptr<const State> observation) const {
        shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::get_child_node_itfc(obsv_itfc);
        return static_pointer_cast<UctDNode>(new_child);
    }
}

/**
 * Boilerplate ThtsCNode interface implementation. Copied from thts_chance_node_template.h.
 */
namespace thts {
    void UctCNode::visit_itfc(ThtsEnvContext& ctx) {
        ThtsEnvContext& ctx_itfc = (ThtsEnvContext&) ctx;
        visit(ctx_itfc);
    }

    shared_ptr<const Observation> UctCNode::sample_observation_itfc(ThtsEnvContext& ctx) {
        ThtsEnvContext& ctx_itfc = (ThtsEnvContext&) ctx;
        shared_ptr<const State> obsv = sample_observation(ctx_itfc);
        return static_pointer_cast<const Observation>(obsv);
    }

    void UctCNode::backup_itfc(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsEnvContext& ctx) 
    {
        ThtsEnvContext& ctx_itfc = (ThtsEnvContext&) ctx;
        backup(
            trial_rewards_before_node, 
            trial_rewards_after_node, 
            trial_cumulative_return_after_node, 
            trial_cumulative_return, 
            ctx_itfc);
    }

    shared_ptr<ThtsDNode> UctCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation, shared_ptr<const State> next_state) const 
    {
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<UctDNode> child_node = create_child_node_helper(obsv_itfc);
        return static_pointer_cast<ThtsDNode>(child_node);
    }
}