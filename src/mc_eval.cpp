#include "mc_eval.h"

#include <cmath>

using namespace std;

/**
 * Eval policy implementation
*/
namespace thts {
    EvalPolicy::EvalPolicy(shared_ptr<ThtsDNode> root_node, shared_ptr<ThtsEnv> thts_env, RandManager& rand_manager) :
        root_node(root_node), cur_node(root_node), thts_env(thts_env), rand_manager(rand_manager) {}
    
    /**
     * Resets cur_node back to root node.
    */
    void EvalPolicy::reset() {
        cur_node = root_node;
    }

    /**
     * Gets a uniform random action.
    */
    shared_ptr<const Action> EvalPolicy::get_random_action(shared_ptr<const State> state) {
        shared_ptr<ActionVector> actions = thts_env->get_valid_actions_itfc(state);
        int indx = rand_manager.get_rand_int(0, actions->size());
        return actions->at(indx);
    }

    /**
     * Gets the best recommendation from the current node.
    */
    shared_ptr<const Action> EvalPolicy::get_action(shared_ptr<const State> state, ThtsEnvContext& context) {
        if (cur_node == nullptr) return get_random_action(state);
        return cur_node->recommend_action_itfc(context);
    }

    /**
     * Updates 'cur_node' for the last step taken in a trial.
    */
    void EvalPolicy::update_step(shared_ptr<const Action> action, shared_ptr<const Observation> obsv) {
        if (cur_node == nullptr) return;
        if (!cur_node->has_child_node_itfc(action)) {
            cur_node = nullptr;
            return;
        }
        ThtsCNode& chance_node = *cur_node->get_child_node_itfc(action);
        if (!chance_node.has_child_node_itfc(obsv)) {
            cur_node = nullptr;
            return;
        }
        cur_node = chance_node.get_child_node_itfc(obsv);
    }
}

/**
 * MC Eval implementation
*/
namespace thts {
    MCEvaluator::MCEvaluator(
        shared_ptr<ThtsEnv> thts_env, shared_ptr<EvalPolicy> policy, int max_trial_length, RandManager& rand_manager) :
            thts_env(thts_env), 
            policy(policy), 
            max_trial_length(max_trial_length), 
            sampled_returns(), 
            rand_manager(rand_manager) {}

    /**
     * Runs a single rollout and stores the result in 'sampled_returns'.
    */
    void MCEvaluator::run_rollout() {
        // Reset
        policy->reset();

        // Bookkeeping
        int num_actions_taken = 0;
        double sample_return = 0.0;
        shared_ptr<const State> state = thts_env->get_initial_state_itfc();
        ThtsEnvContext& context = *thts_env->sample_context_itfc(state);

        // Run trial
        while (num_actions_taken < max_trial_length && !thts_env->is_sink_state_itfc(state)) {
            shared_ptr<const Action> action = policy->get_action(state, context);
            shared_ptr<const State> next_state = thts_env->sample_transition_distribution_itfc(
                state, action, rand_manager);
            shared_ptr<const Observation> obsv = thts_env->sample_observation_distribution_itfc(
                action, next_state, rand_manager);
            
            sample_return += thts_env->get_reward_itfc(state, action, obsv);

            policy->update_step(action, obsv);
            state = next_state;
        }

        // store
        sampled_returns.push_back(sample_return);
    }
    
    /**
     * Run 'num_rollout' many rollouts to gather stats.
    */
    void MCEvaluator::run_rollouts(int num_rollouts) {
        for (int i=0; i < num_rollouts; i++) {
            run_rollout();
        }
    }

    /**
     * Returns the mean return of 'sampled_returns'
    */
    double MCEvaluator::get_mean_return() {
        double weight = 1.0 / sampled_returns.size();
        double mean = 0.0;
        for (double val : sampled_returns) {
            mean += weight * val;
        }
        return mean;
    }

    /**
    * Returns the stddev of 'sampled_returns'
    */
    double MCEvaluator::get_stddev_return() {
        double mean = get_mean_return();
        double weight = 1.0 / (sampled_returns.size() - 1.0);
        double stddev = 0.0;
        for (double val : sampled_returns) {
            stddev += weight * pow(val - mean, 2.0);
        }
        return stddev;
    }
}