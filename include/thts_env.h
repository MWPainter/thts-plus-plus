#pragma once

#include "thts_env_context.h"
#include "thts_manager.h"
#include "thts_types.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace thts {
   /**
     * An abstract class for representing an environment.
     * 
     * Defines a set of functions that should be general enough to handle a range of environments for planning, like 
     * MDPs and POMDPs. 
     * 
     * Additionally, any subclasses that wish to plan using a transposition table should implement the std::hash and 
     * std::equal_to functions. 
     * 
     * Uses the State, Action and Observation objects from thts_types.h, as base types for the following:
     *      State: Objects representing the current state of the world
     *      Action: Objects representing the actions that an agent can take
     *      Observation: 
     *          Objects representing the outcomes that can occur from taking actions in the environment. For something 
     *          like an MDP, we would probably want Observation's to be the successor states (which would require 
     *          Observation == State subtypes)
     * 
     * Attributes:
     *      env_id: A sting id for the environment.
     */
    class ThtsEnv {

        protected:
            std::string env_id;

        public:
            /**
             * Default destructor is sufficient. But need to declare it virtual.
             */
            virtual ~ThtsEnv() = default;

            /**
             * Returns the initial state for the environment.
             * 
             * Args:
             *      None
             * 
             * Returns:
             *      Initial state for this environment instance
             */
            virtual std::shared_ptr<const State> get_initial_state_itfc() const = 0;

            /**
             * Returns if a state is a sink state.
             * 
             * Args:
             *      state: The state to be checked if it is a sink state
             * 
             * Returns:
             *      True if 'state' is a sink state and false otherwise
             */
            virtual bool is_sink_state_itfc(std::shared_ptr<const State> state) const = 0;

            /**
             * Returns a list of actions that are valid in a given state.
             * 
             * Args:
             *      state: The state that we want a list of available actions from
             * 
             * Returns:
             *      Returns a list of actions available from 'state'
             */
            virtual std::shared_ptr<ActionVector> get_valid_actions_itfc(
                std::shared_ptr<const State> state) const = 0;

            /**
             * Returns a distribution over observations from a state action pair.
             * 
             * Given a state and action returns a distribution of possible observations. The probability distribution 
             * is returned in the form of a map, where the keys are of the Observation type, and the values are 
             * doubles, which sum to one.
             * 
             * Args:
             *      state: The state to get a transition distribution from
             *      action: The action to get a transition distribution for
             * 
             * Returns:
             *      Returns a distribution over observation from taking 'action' in state 'state'.
             */
            virtual std::shared_ptr<ObservationDistr> get_transition_distribution_itfc(
                std::shared_ptr<const State> state, std::shared_ptr<const Action> action) const = 0;

            /**
             * Samples an observation when taking an action from a state.
             * 
             * Given a state, action pair, samples a possible observation that can arrise.
             * 
             * Args:
             *      state: The state to sample an observation from
             *      action: The action taken to sample an observation for
             *      thts_manager: A pointer to the thts_manager to access the random number sampling interface
             * 
             * Returns:
             *      Returns an observation sampled from taking 'action' from 'state'
             */
            virtual std::shared_ptr<const Observation> sample_transition_distribution_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action, 
                std::shared_ptr<ThtsManager> thts_manager) const = 0;
            
            /**
             * Returns the reward for a given state, action, observation tuple.
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
            virtual double get_reward_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action, 
                std::shared_ptr<const Observation> observation=nullptr) const = 0;

            /**
             * Samples a context that can be used to store information throughout a single trial.
             * 
             * Sometimes it is useful to place each trial in some sort of context, or a context can be used to cache 
             * information that doesn't need to be stored in the tree search permenantly, but is useful computationally. 
             * This function generates a context to be used. Most of the time it will be something like an empty map. 
             * 
             * Args:
             *      state: The initial state
             * 
             * Returns:
             *      A ThtsEnvContext object, that will be passed to the Thts functions for a single trial, used to 
             *      provide some context or space for caching.
             */
            virtual ThtsEnvContext sample_context_itfc(std::shared_ptr<const State> state) const;
    };
}