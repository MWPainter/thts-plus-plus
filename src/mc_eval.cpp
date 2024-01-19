#include "mc_eval.h"

#include <cmath>
#include <thread>

using namespace std;

/**
 * Eval policy implementation
*/
namespace thts {
    EvalPolicy::EvalPolicy(
        shared_ptr<const ThtsDNode> root_node, 
        shared_ptr<ThtsEnv> thts_env,
        shared_ptr<ThtsManager> manager) :
            root_node(root_node), 
            cur_node(root_node),
            thts_env(thts_env),
            manager(manager) {}

    EvalPolicy::EvalPolicy(const EvalPolicy& policy, shared_ptr<ThtsEnv> thts_env) :
        root_node(policy.root_node), 
        cur_node(policy.root_node), 
        thts_env(thts_env),
        manager(policy.manager) {}
    
    /**
     * Resets cur_node back to root node.
    */
    void EvalPolicy::reset() {
        cur_node = root_node;
    }

    /**
     * Gets a uniform random action.
    */
    shared_ptr<const Action> EvalPolicy::get_random_action(
        shared_ptr<const State> state, ThtsEnvContext& ctx) 
    {
        shared_ptr<ActionVector> actions = thts_env->get_valid_actions_itfc(state, ctx);
        int indx = manager->get_rand_int(0, actions->size());
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
        shared_ptr<EvalPolicy> policy, 
        int max_trial_length, 
        shared_ptr<ThtsManager> manager) : 
            policy(policy), 
            max_trial_length(max_trial_length), 
            sampled_returns(), 
            manager(manager),
            lock(),
            num_rollouts_to_run(0),
            num_rollouts_started(0),
            rollouts_lock()
    {
    }

    /**
     * When getting things from the tree it calls manager->thts_env()
     * To get correct env for it, it needs to have the std::this_thread::get_id() mapped to 'thread_id'
    */
    void MCEvaluator::setup_thread(int thread_id) 
    {
        manager->register_thread_id(thread_id);
    }

    /**
     * Runs a single rollout and stores the result in 'sampled_returns'.
    */
    void MCEvaluator::run_rollout(int thread_id, EvalPolicy& thread_policy) {
        // Reset
        shared_ptr<ThtsEnv> thts_env = manager->thts_env(thread_id);
        thread_policy.reset();
        thts_env->reset_itfc();

        // Bookkeeping
        int num_actions_taken = 0;
        double sample_return = 0.0;
        shared_ptr<ThtsEnvContext> context = thts_env->sample_context_itfc(thread_id, *manager);
        manager->register_thts_context(thread_id, context);
        shared_ptr<const State> state = thts_env->get_initial_state_itfc();

        // Run trial
        while (num_actions_taken++ < max_trial_length && !thts_env->is_sink_state_itfc(state, *context)) {
            shared_ptr<const Action> action = thread_policy.get_action(state, *context);
            shared_ptr<const State> next_state = thts_env->sample_transition_distribution_itfc(
                state, action, *manager, *context);
            shared_ptr<const Observation> obsv = static_pointer_cast<const Observation>(next_state); //TODO: do this properly for partial obs at some point, need to be careful with PythonGym envs and not calling step twice
            
            sample_return += thts_env->get_reward_itfc(state, action, *context);

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
        int total_rollouts, int thread_id, int num_threads, shared_ptr<EvalPolicy> thread_policy) 
    {
        setup_thread(thread_id);
        lock_guard<mutex> lg(rollouts_lock);
        while (num_rollouts_started++ < num_rollouts_to_run) {
            rollouts_lock.unlock();
            run_rollout(thread_id, *thread_policy);
            rollouts_lock.lock();
        }
    }

    /**
     * Runs 'num_rollouts' using 'num_threads'. Just sets each thread up, starts it running and then waits for them. 
     * Note that a pointer to a copy constructed policy is passed to each thread for it to use. (So each thread can 
     * assume that it has it's own thread_policy, copied from 'policy')
    */
    void MCEvaluator::run_rollouts(int num_rollouts, int num_threads) {
        // setup vars
        num_rollouts_to_run = num_rollouts;
        num_rollouts_started = 0;

        // spawn
        vector<thread> threads;
        for (int i=0; i<num_threads; i++) {
            shared_ptr<EvalPolicy> thread_eval_policy = make_shared<EvalPolicy>(*policy,manager->thts_env(i));
            threads.push_back(thread(
                &MCEvaluator::thread_run_rollouts, 
                this, 
                num_rollouts, 
                i, 
                num_threads, 
                thread_eval_policy));
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