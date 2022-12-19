#pragma once

#include "thts_chance_node.h"
#include "thts_decision_node.h"
#include "thts_env.h"
#include "thts_env_context.h"
#include "thts_manager.h"
#include "thts_types.h"

#include <memory>
#include <vector>

namespace thts {
    /**
     * Converts a thts tree into a complete policy that can be used for evaluation.
     * 
     * While performing a trial will trace the trial in the tree and use the recommendations given by decision nodes.
     * When a trial reaches a state that there isn't a corresponding tree node, uniform random policy is used.
     * 
     * N.B. Currently only works for fully observable environments.
     * 
     * Member variables:
     *      root_node: The root node of the thts tree to use
     *      cur_node: The current decision node to use to make recommendations in the policy (iff null, use uniform)
     *      thts_env: The environment for when random actions need to be sampled
     *      rand_manager: Manager for rng
    */
    class EvalPolicy {
        protected:
            std::shared_ptr<ThtsDNode> root_node;
            std::shared_ptr<ThtsDNode> cur_node;
            std::shared_ptr<ThtsEnv> thts_env;
            RandManager& rand_manager;

        public:
            EvalPolicy(
                std::shared_ptr<ThtsDNode> root_node, std::shared_ptr<ThtsEnv> thts_env, RandManager& rand_manager);

            /**
             * Resets cur_node back to root node.
            */
            void reset();

            /**
             * Gets a uniform random action.
            */
            std::shared_ptr<const Action> get_random_action(std::shared_ptr<const State> state);

            /**
             * Gets the best recommendation from the current node.
            */
            std::shared_ptr<const Action> get_action(std::shared_ptr<const State> state, ThtsEnvContext& context);

            /**
             * Updates 'cur_node' for the last step taken in a trial.
            */
           void update_step(std::shared_ptr<const Action> action, std::shared_ptr<const Observation> obsv);
    };

    /**
     * MC Evaluator
     * 
     * N.B. Currently only works for fully observable environments.
     * 
     * Member variables:
     *      thts_env: The env that we want to evaluate in
     *      policy: The policy to evaluate
     *      max_trial_length: The maximum trial length to use in MC evaluations
     *      sampled_returns: A list of sampled returns 
     *      rand_manager: Manager for rng
    */
    class MCEvaluator {
        protected:
            std::shared_ptr<ThtsEnv> thts_env;
            std::shared_ptr<EvalPolicy> policy;
            int max_trial_length;
            std::vector<double> sampled_returns;
            RandManager& rand_manager;

            /**
             * Runs a single rollout and stores the result in 'sampled_returns'.
            */
            void run_rollout();

        public:
            MCEvaluator(
                std::shared_ptr<ThtsEnv> thts_Env,
                std::shared_ptr<EvalPolicy> eval_policy,
                int max_trial_length,
                RandManager& rand_manager);

            /**
             * Run 'num_rollout' many rollouts to gather stats.
            */
            void run_rollouts(int num_rollouts);

            /**
             * Returns the mean return of 'sampled_returns'
            */
           double get_mean_return();

           /**
            * Returns the stddev of 'sampled_returns'
           */
          double get_stddev_return();
    };
}