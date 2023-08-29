#pragma once

#include "thts_env.h"
#include "thts_env_context.h"
#include "thts_types.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

namespace thts {
    // Forward declare
    class RandManager;
    
    /**
     * An abstract class for representing an multi-objective environment.
     * 
     * Member variables:
     *      reward_dim: 
     */
    class MoThtsEnv : public ThtsEnv {
        protected:
            int reward_dim;

        public:
            /**
             * Constructor
             */
            MoThtsEnv(int reward_dim, bool is_fully_observable);

            /**
             * Reward dim getter
            */
            int get_reward_dim();
            
            /**
             * Override ThtsEnv get_reward_itfc to make it throw an exception. It shouldn't be called by multi-objective
             * implementations.
             */
            virtual double get_reward_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action, 
                std::shared_ptr<const Observation> observation=nullptr) const;
            
            /**
             * Returns the multi objective reward for a given state, action, observation tuple.
             * 
             * Commonly the reward is written as a function of just the state and action pair. But we provide the 
             * option to depend on the observation too. 
             * 
             * Args:
             *      state: The current state to get a reward for
             *      action: The action taken to get a reward for
             *      observation: 
             *          The (optional) observation sampled from the state, action pair that can optionally be used as 
             *          part of the reward function.
             * 
             * Returns:
             *      The reward for taking 'action' from 'state' (and sampling 'observation')
             */
            virtual Eigen::VectorXd get_mo_reward_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action, 
                std::shared_ptr<const Observation> observation=nullptr) const = 0;
    };
}