#pragma once

#include "mo/mo_thts_env.h"
#include "thts_manager.h"
#include "thts_types.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

namespace thts{

    /** 
     * Parameterised toy environment to check scalability of algorithms
     * 
     * Uses 'num_actions' many, with dimension of 'num_rewards'
     * There is one action per reward, and following action i gives a reward of 'reward_vectors[i] / tree_depth'
     * 'tree_depth' is how many steps are taken in the environment, and every path has same length (balanced tree)
     * 
     * 'sparse' option is used to give all of the accumulated rewards at the end of the trial
     * 
     * state space is a list of length 'num_actions + 1', where the first element is how many steps have been taken 
     * so far
     * 
     * N.B. this class needs to call some python, and needs the python interpreter to exist already.
     */
    class ToyTreeEnv : public MoThtsEnv {

        /**
         * Core ToyTreeEnv implementaion.
         */
        public:
            int num_actions;
            int tree_depth;
            bool sparse;
            std::vector<Eigen::ArrayXd> reward_vectors;



        /**
         * Core ThtsEnv implementation functinos.
         */
        public:
            /**
             * Constructor
             */
            ToyTreeEnv(int num_rewards, int num_actions, int tree_depth, bool sparse);
            ToyTreeEnv(const ToyTreeEnv& other);
            virtual std::shared_ptr<ThtsEnv> clone();

            /**
             * Loading rewards from cached
             * Uses util/generate_well_spaced_vectors.py to generate and util/cached_hypersphere_points/.* for cache
             */
            void load_cached_rewards(int num_rewards, int num_actions);
            void generate_cached_rewards(int num_rewards, int num_actions);

            /**
             * Mark destructor as virtual for subclassing.
             */
            virtual ~ToyTreeEnv() = default;

            /**
             * Returns the initial state for the environment.
             * 
             * Returns:
             *      Initial state for this environment instance
             */
            std::shared_ptr<const IntVectorState> get_initial_state() const;

            /**
             * Returns if a state is a sink state.
             * 
             * Args:
             *      state: The state to be checked if it is a sink state
             * 
             * Returns:
             *      True if 'state' is a sink state and false otherwise
             */
            bool is_sink_state(std::shared_ptr<const IntVectorState> state) const;

            /**
             * Returns a list of actions that are valid in a given state.
             * 
             * Args:
             *      state: The state that we want a list of available actions from
             * 
             * Returns:
             *      Returns a list of actions available from 'state'
             */
            std::shared_ptr<IntActionVector> get_valid_actions(std::shared_ptr<const IntVectorState> state) const;

            /**
             * Returns next state for taking given action
             */
            std::shared_ptr<IntVectorState> get_next_state(
                std::shared_ptr<const IntVectorState> state, std::shared_ptr<const IntAction> action) const;

            /**
             * Returns a distribution over successor states from a state action pair.
             * 
             * Given a state and action returns a distribution of possible successor states. The probability 
             * distribution is returned in the form of a map, where the keys are of the State type, and the values are 
             * doubles, which sum to one.
             * 
             * Args:
             *      state: The state to get a transition distribution from
             *      action: The action to get a transition distribution for
             * 
             * Returns:
             *      Returns a successor state distribution from taking 'action' in state 'state'.
             */
            std::shared_ptr<IntVectorStateDistr> get_transition_distribution(
                std::shared_ptr<const IntVectorState> state, std::shared_ptr<const IntAction> action) const;

            /**
             * Samples an successor state when taking an action from a state.
             * 
             * Given a state, action pair, samples a possible successor state that can arrise.
             * 
             * Args:
             *      state: The state to sample an observation from
             *      action: The action taken to sample an observation for
             *      rand_manager: A RandManager ref to access the random number sampling interface
             * 
             * Returns:
             *      Returns an successor state sampled from taking 'action' from 'state'
             */
            std::shared_ptr<const IntVectorState> sample_transition_distribution(
                std::shared_ptr<const IntVectorState> state, 
                std::shared_ptr<const IntAction> action, 
                RandManager& rand_manager) const;
            
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
            Eigen::ArrayXd get_mo_reward(
                std::shared_ptr<const IntVectorState> state, 
                std::shared_ptr<const IntAction> action) const;



        /**
         * ThtsEnv interface function definitions. Boilerplate implementations provided from thts_env_template.h
         */
        public:
            virtual std::shared_ptr<const State> get_initial_state_itfc() const;
            virtual bool is_sink_state_itfc(std::shared_ptr<const State> state, ThtsEnvContext& ctx) const override;
            virtual std::shared_ptr<ActionVector> get_valid_actions_itfc(
                std::shared_ptr<const State> state,
                ThtsEnvContext& ctx) const override;
            virtual std::shared_ptr<StateDistr> get_transition_distribution_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action,
                ThtsEnvContext& ctx) const override;
            virtual std::shared_ptr<const State> sample_transition_distribution_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action, 
                 RandManager& rand_manager,
                ThtsEnvContext& ctx) const override;
            virtual Eigen::ArrayXd get_mo_reward_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action,
                ThtsEnvContext& ctx) const override;
    };
}
