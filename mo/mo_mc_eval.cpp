#include "mo/mo_mc_eval.h"

#include "mo/mo_helper.h"

#include <iostream>

using namespace std;

/**
 * MC Eval implementation
*/
namespace thts {
    MoMCEvaluator::MoMCEvaluator(
        shared_ptr<EvalPolicy> policy, 
        int max_trial_length, 
        shared_ptr<MoThtsManager> manager,
        Vec r_min,
        Vec r_max,
        bool well_spaced_eval
        bool normalised_value_space) :
            MCEvaluator(policy,max_trial_length,manager),
            mo_sampled_returns(),
            sampled_ctx_returns(),
            sampled_normalised_ctx_returns(),
            r_min(r_min),
            r_max(r_max),
            well_spaced_eval(well_spaced_eval),
            normalised_value_space(normalised_value_space),
            context_weights()
    {
    }

    /**
     * Computes a reweighting of the contextual return 
     * alpha(w) = 1 / dot(r_max - r_min, w)
     */
    double MoMCEvaluator::reweighting_coefficient(Vec context_weight) 
    {
        return 1.0 / context_weight.dot(r_max - r_min);
    }

    /**
     * Runs a single rollout and stores the result in 'sampled_returns'.
    */
    void MoMCEvaluator::run_rollout(int thread_id, EvalPolicy& thread_policy) {
        // Get context for this rollout
        lock.lock();
        Vec context_vec = context_weights.back();
        context_weights.pop_back();
        lock.unlock();

        // If normalised value space, then scale the context weight by (r_max - r_min) and normalise
        // See comments in mo_mc_eval.h for more details (but gives algorithm correct unnormalised weights)
        Vec context_vec_for_alg = Vec(context_vec);
        if (normalised_value_space) {
            context_vec_for_alg = context_vec_for_alg * (r_max - r_min);
            context_vec_for_alg = context_vec_for_alg / context_vec_for_alg.norm();
        }

        // Reset
        shared_ptr<MoThtsEnv> thts_env = dynamic_pointer_cast<MoThtsEnv>(manager->thts_env(thread_id));
        thread_policy.reset();
        thts_env->reset_itfc();

        // Bookkeeping
        int num_actions_taken = 0;
        Vec mo_sample_return = Vec(thts_env->get_reward_dim(), 0.0);

        // Start trial
        // shared_ptr<MoThtsContext> mo_context = static_pointer_cast<MoThtsContext>(
        //     thts_env->sample_context_itfc(thread_id, *manager));
        shared_ptr<MoThtsContext> mo_context = make_shared<MoThtsContext>(context_vec_for_alg);
        manager->register_thts_context(thread_id, mo_context);
        shared_ptr<const State> state = thts_env->get_initial_state_itfc();

        // Run trial
        while (num_actions_taken++ < max_trial_length && !thts_env->is_sink_state_itfc(state, *mo_context)) {
            shared_ptr<const Action> action = thread_policy.get_action(state, *mo_context);
            shared_ptr<const State> next_state = thts_env->sample_transition_distribution_itfc(
                state, action, *manager, *mo_context);
            shared_ptr<const Observation> obsv = static_pointer_cast<const Observation>(next_state); //TODO: do this properly for partial obs at some point, need to be careful with PythonGym envs and not calling step twice
            
            mo_sample_return += thts_env->get_mo_reward_itfc(state, action, *mo_context);

            thread_policy.update_step(action, obsv);
            state = next_state;
        }

        // If normalised value space, then normalise the return from the algorithm between 0 and 1
        if (normalised_value_space) {
            mo_sample_return = (mo_sample_return - r_min) / (r_max - r_min);
        }

        // store rollout result
        double contextual_return = mo_sample_return.dot(context_vec);
        double reweighted_contextual_return = contextual_return * reweighting_coefficient(context_vec);
        lock_guard lg(lock);
        mo_sampled_returns.push_back(mo_sample_return);
        sampled_ctx_returns.push_back(contextual_return);
        sampled_reweighted_ctx_returns.push_back(reweighted_contextual_return);
    }

    /**
     * Fill out context weights, and then call MCEvaluator version
     */
    void MoMCEvaluator::run_rollouts(int num_rollouts, int num_threads) 
    {
        context_weights.clear();
        context_weights.reserve(num_rollouts);
        int dim = r_max.vec.size();
        if (well_spaced_eval) {
            vector<Eigen::ArrayXd> well_spaced_vectors = thts::helper::get_well_spaced_simplex_points(num_rollouts,dim);
            for (Eigen::ArrayXd& vec_arr : well_spaced_vectors) {
                context_weights.push_back(vec_arr);
            }
        } else {
            for (int i=0; i<num_rollouts; i++) {
                RandManager rand_manager;
                context_weights.push_back(thts::helper::sample_uniform_random_simplex_vector(rand_manager,dim));
            }
        }

        MCEvaluator::run_rollouts(num_rollouts, num_threads);
    }

    Vec MoMCEvaluator::get_mo_return_mean() 
    {
        shared_ptr<MoThtsEnv> thts_env = dynamic_pointer_cast<MoThtsEnv>(manager->thts_env());
        int reward_dim = thts_env->get_reward_dim();
        double weight = 1.0 / mo_sampled_returns.size();
        Vec mean = Vec(reward_dim, 0.0);
        for (Vec val : mo_sampled_returns) {
            mean += weight * val;
        }
        return mean;

    }

    double MoMCEvaluator::get_mo_ctx_return_mean()
    {
        double weight = 1.0 / sampled_ctx_returns.size();
        double mean = 0.0;
        for (double val : sampled_ctx_returns) {
            mean += weight * val;
        }
        return mean;
    }
    
    double MoMCEvaluator::get_reweighted_mo_ctx_return_mean()
    {
        double weight = 1.0 / sampled_reweighted_ctx_returns.size();
        double mean = 0.0;
        for (double val : sampled_reweighted_ctx_returns) {
            mean += weight * val;
        }
        return mean;
    }
    
    Vec MoMCEvaluator::get_mo_return_variance()
    {
        shared_ptr<MoThtsEnv> thts_env = dynamic_pointer_cast<MoThtsEnv>(manager->thts_env());
        double reward_dim = thts_env->get_reward_dim();
        Vec mean = get_mo_return_mean();
        double weight = 1.0 / mo_sampled_returns.size();
        Vec stddev = Vec(reward_dim, 0.0);
        for (Vec val : mo_sampled_returns) {
            Vec diff = val - mean;
            stddev += weight * (diff * diff);
        }
        return stddev;

    }
    
    double MoMCEvaluator::get_mo_return_variance(Vec context_weights)
    {
        return context_weights.dot(get_mo_return_variance());
    }
    
    double MoMCEvaluator::get_mo_ctx_return_variance()
    {
        double mean = get_mo_ctx_return_mean();
        double weight = 1.0 / (sampled_ctx_returns.size() - 1.0);
        double stddev = 0.0;
        for (double val : sampled_ctx_returns) {
            stddev += weight * pow(val - mean, 2.0);
        }
        return stddev;
    }
    
    double MoMCEvaluator::get_reweighted_mo_ctx_return_variance()
    {
        double mean = get_reweighted_mo_ctx_return_mean();
        double weight = 1.0 / (sampled_reweighted_ctx_returns.size() - 1.0);
        double stddev = 0.0;
        for (double val : sampled_reweighted_ctx_returns) {
            stddev += weight * pow(val - mean, 2.0);
        }
        return stddev;
    }
    
}