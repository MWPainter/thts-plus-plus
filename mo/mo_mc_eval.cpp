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
        Eigen::ArrayXd r_min,
        Eigen::ArrayXd r_max) :
            MCEvaluator(policy,max_trial_length,manager),
            mo_sampled_returns(),
            sampled_ctx_returns(),
            sampled_normalised_ctx_returns(),
            r_min(r_min),
            r_max(r_max)
    {
    }

    /**
     * Runs a single rollout and stores the result in 'sampled_returns'.
    */
    void MoMCEvaluator::run_rollout(int thread_id, EvalPolicy& thread_policy) {
        // Reset
        shared_ptr<MoThtsEnv> thts_env = dynamic_pointer_cast<MoThtsEnv>(manager->thts_env(thread_id));
        thread_policy.reset();
        thts_env->reset_itfc();

        // Bookkeeping
        int num_actions_taken = 0;
        Eigen::ArrayXd mo_sample_return = Eigen::ArrayXd::Zero(thts_env->get_reward_dim());
        // MoThtsContext& context = (MoThtsContext&) *thts_env.sample_context_itfc(thread_id, *manager);
        shared_ptr<MoThtsContext> mo_context = static_pointer_cast<MoThtsContext>(
            thts_env->sample_context_itfc(thread_id, *manager));
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

        // store
        double contextual_return = thts::helper::dot(mo_context->context_weight, mo_sample_return);
        Eigen::ArrayXd normalised_sample_return = (mo_sample_return - r_min) / (r_max - r_min);
        double normalised_contextual_return = thts::helper::dot(mo_context->context_weight, normalised_sample_return);
        lock_guard lg(lock);
        mo_sampled_returns.push_back(mo_sample_return);
        sampled_ctx_returns.push_back(contextual_return);
        sampled_normalised_ctx_returns.push_back(normalised_contextual_return);
    }

    Eigen::ArrayXd MoMCEvaluator::get_mean_mo_return() 
    {
        shared_ptr<MoThtsEnv> thts_env = dynamic_pointer_cast<MoThtsEnv>(manager->thts_env());
        int reward_dim = thts_env->get_reward_dim();
        double weight = 1.0 / mo_sampled_returns.size();
        Eigen::ArrayXd mean = Eigen::ArrayXd::Zero(reward_dim);
        for (Eigen::ArrayXd val : mo_sampled_returns) {
            mean += weight * val;
        }
        return mean;

    }

    double MoMCEvaluator::get_mean_mo_return(Eigen::ArrayXd context_weights)
    {
        return thts::helper::dot(context_weights, get_mean_mo_return());
    }

    double MoMCEvaluator::get_mean_mo_ctx_return()
    {
        double weight = 1.0 / sampled_ctx_returns.size();
        double mean = 0.0;
        for (double val : sampled_ctx_returns) {
            mean += weight * val;
        }
        return mean;
    }
    
    double MoMCEvaluator::get_mean_mo_normalised_ctx_return()
    {
        double weight = 1.0 / sampled_normalised_ctx_returns.size();
        double mean = 0.0;
        for (double val : sampled_normalised_ctx_returns) {
            mean += weight * val;
        }
        return mean;
    }
    
    Eigen::ArrayXd MoMCEvaluator::get_stddev_mo_return()
    {
        shared_ptr<MoThtsEnv> thts_env = dynamic_pointer_cast<MoThtsEnv>(manager->thts_env());
        double reward_dim = thts_env->get_reward_dim();
        Eigen::ArrayXd mean = get_mean_mo_return();
        double weight = 1.0 / mo_sampled_returns.size();
        Eigen::ArrayXd stddev = Eigen::ArrayXd::Zero(reward_dim);
        for (Eigen::ArrayXd val : mo_sampled_returns) {
            stddev += weight * (val - mean).pow(2.0);
        }
        return stddev;

    }
    
    double MoMCEvaluator::get_stddev_mo_return(Eigen::ArrayXd context_weights)
    {
        return thts::helper::dot(context_weights, get_stddev_mo_return());
    }
    
    double MoMCEvaluator::get_stddev_mean_mo_ctx_return()
    {
        double mean = get_mean_mo_ctx_return();
        double weight = 1.0 / (sampled_ctx_returns.size() - 1.0);
        double stddev = 0.0;
        for (double val : sampled_ctx_returns) {
            stddev += weight * pow(val - mean, 2.0);
        }
        return stddev;
    }
    
    double MoMCEvaluator::get_stddev_mean_mo_normalised_ctx_return()
    {
        double mean = get_mean_mo_normalised_ctx_return();
        double weight = 1.0 / (sampled_normalised_ctx_returns.size() - 1.0);
        double stddev = 0.0;
        for (double val : sampled_normalised_ctx_returns) {
            stddev += weight * pow(val - mean, 2.0);
        }
        return stddev;
    }
    
}