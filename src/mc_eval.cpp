#include "mc_eval.h"

#include <cmath>
#include <thread>

using namespace std;

/**
 * Eval policy implementation
*/
namespace thts {
    EvalPolicy::EvalPolicy(
        shared_ptr<const ThtsDNode> root_node, shared_ptr<const ThtsEnv> thts_env, RandManager& rand_manager) :
            root_node(root_node), cur_node(root_node), thts_env(thts_env), rand_manager(rand_manager) {}

    EvalPolicy::EvalPolicy(const EvalPolicy& policy) :
        root_node(policy.root_node), 
        cur_node(policy.root_node), 
        thts_env(policy.thts_env), 
        rand_manager(policy.rand_manager) {}
    
    /**
     * Resets cur_node back to root node.
    */
    void EvalPolicy::reset() {
        cur_node = root_node;
    }

    /**
     * Gets a uniform random action.
    */
    shared_ptr<const Action> EvalPolicy::get_random_action(shared_ptr<const State> state, ThtsEnvContext& ctx) {
        shared_ptr<ActionVector> actions = thts_env->get_valid_actions_itfc(state, ctx);
        int indx = rand_manager.get_rand_int(0, actions->size());
        return actions->at(indx);
    }

    /**
     * Gets the best recommendation from the current node.
    */
    shared_ptr<const Action> EvalPolicy::get_action(shared_ptr<const State> state, ThtsEnvContext& context) {
        if (cur_node == nullptr) return get_random_action(state,context);
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
        int num_envs,
        shared_ptr<ThtsEnv> thts_env, 
        EvalPolicy& policy, 
        int max_trial_length, 
        RandManager& rand_manager) :
            num_envs(num_envs),
            thts_envs(), 
            policy(policy), 
            max_trial_length(max_trial_length), 
            sampled_returns(), 
            rand_manager(rand_manager),
            lock() 
    {
        if (num_envs < 1) {
            throw runtime_error("Shouldnt try to run an MCEvaluator with < 1 env");
        }
        thts_envs.push_back(thts_env);
        for (int i=1; i<num_envs; i++) {
            thts_envs.push_back(thts_env->clone());
        }
    }
    
    /**
     * Gets appropriate thts env for this thread
    */
    shared_ptr<ThtsEnv> MCEvaluator::get_env(int thread_id) {
        return thts_envs[thread_id % num_envs];
    }

    /**
     * Empty setup thread method, it's for subclasses really
    */
    void MCEvaluator::setup_thread(int thread_id) 
    {
    }

    /**
     * Runs a single rollout and stores the result in 'sampled_returns'.
    */
    void MCEvaluator::run_rollout(int thread_id, EvalPolicy& thread_policy) {
        // Reset
        thread_policy.reset();

        // Bookkeeping
        int num_actions_taken = 0;
        double sample_return = 0.0;
        shared_ptr<ThtsEnv> thts_env = get_env(thread_id);
        ThtsEnvContext& context = *thts_env->sample_context_and_reset_itfc(thread_id);
        shared_ptr<const State> state = thts_env->get_initial_state_itfc();

        // Run trial
        while (num_actions_taken < max_trial_length && !thts_env->is_sink_state_itfc(state, context)) {
            shared_ptr<const Action> action = thread_policy.get_action(state, context);
            shared_ptr<const State> next_state = thts_env->sample_transition_distribution_itfc(
                state, action, rand_manager, context);
            shared_ptr<const Observation> obsv = thts_env->sample_observation_distribution_itfc(
                action, next_state, rand_manager, context);
            
            sample_return += thts_env->get_reward_itfc(state, action, context);

            thread_policy.update_step(action, obsv);
            state = next_state;
        }

        // store
        lock_guard lg(lock);
        sampled_returns.push_back(sample_return);
    }
    
    /**
     * Called as a thread. Runs this threads portion of 'total_rollouts' many rollouts. To make coding simple as we 
     * know exactly how many rollouts to perform ahead of time in this case, this thread will just be allocated all of 
     * the rollouts numbered == thread_id mod num_threads.
    */
    void MCEvaluator::thread_run_rollouts(
        int total_rollouts, int thread_id, int num_threads, unique_ptr<EvalPolicy> thread_policy) 
    {
        setup_thread(thread_id);
        for (int i=thread_id; i < total_rollouts; i+=num_threads) {
            run_rollout(thread_id, *thread_policy);
        }
    }

    /**
     * Runs 'num_rollouts' using 'num_threads'. Just sets each thread up, starts it running and then waits for them. 
     * Note that a pointer to a copy constructed policy is passed to each thread for it to use. (So each thread can 
     * assume that it has it's own thread_policy, copied from 'policy')
    */
    void MCEvaluator::run_rollouts(int num_rollouts, int num_threads) {
        // spawn
        vector<thread> threads;
        for (int i=0; i<num_threads; i++) {
            threads.push_back(thread(
                &MCEvaluator::thread_run_rollouts, 
                this, 
                num_rollouts, 
                i, 
                num_threads, 
                make_unique<EvalPolicy>(policy)));
        }

        // wait
        for (int i=0; i<num_threads; i++) {
            threads[i].join();
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