#pragma once

#include "mo/mo_thts_env.h"
#include "thts_manager.h"
#include "thts_types.h"

#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>

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
            int total_actions;
            int num_xtra_actions;
            double reward_scale;
            int horizon;
            double random_action_prob;
            std::vector<std::pair<double,double>> xtra_action_rewards;
            std::vector<std::pair<int,int>> xtra_reward_dims;

        /**
         * Internal RNG used to sample the random-action substitution inside get_mo_reward.
         * 'get_mo_reward' is const (and may be called concurrently from multiple search
         * threads) so the generator and mutex are mutable.
         */
        private:
            mutable std::mutex rand_action_rng_mutex;
            mutable std::mt19937 rand_action_rng;



        /**
         * Core ThtsEnv implementation functinos.
         */
        public:
            /**
             * Constructor
             *
             * random_action_prob: probability in [0,1] that, when computing a reward,
             * the agent's chosen action is replaced by a uniformly random action drawn
             * from the full action set. Defaults to 0 (deterministic, original behaviour).
             */
            ToyTreeEnv(
                int reward_dim,
                int num_xtra_actions,
                double axis_reward_ratio=0.9,
                double random_action_prob=0.0f);
            ToyTreeEnv(const ToyTreeEnv& other);
            virtual std::shared_ptr<ThtsEnv> clone();

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
            std::shared_ptr<const IntState> get_initial_state() const;

            /**
             * Returns if a state is a sink state.
             * 
             * Args:
             *      state: The state to be checked if it is a sink state
             * 
             * Returns:
             *      True if 'state' is a sink state and false otherwise
             */
            bool is_sink_state(std::shared_ptr<const IntState> state) const;

            /**
             * Returns a list of actions that are valid in a given state.
             * 
             * Args:
             *      state: The state that we want a list of available actions from
             * 
             * Returns:
             *      Returns a list of actions available from 'state'
             */
            std::shared_ptr<IntActionVector> get_valid_actions(std::shared_ptr<const IntState> state) const;

            /**
             * Returns next state for taking given action
             */
            std::shared_ptr<IntState> get_next_state(
                std::shared_ptr<const IntState> state, std::shared_ptr<const IntAction> action) const;

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
            std::shared_ptr<IntStateDistr> get_transition_distribution(
                std::shared_ptr<const IntState> state, std::shared_ptr<const IntAction> action) const;

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
            std::shared_ptr<const IntState> sample_transition_distribution(
                std::shared_ptr<const IntState> state, 
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
                std::shared_ptr<const IntState> state, 
                std::shared_ptr<const IntAction> action) const;



        /**
         * ThtsEnv interface function definitions. Boilerplate implementations provided from thts_env_template.h
         */
        public:
            virtual std::shared_ptr<const State> get_initial_state_itfc() const;
            virtual bool is_sink_state_itfc(std::shared_ptr<const State> state, ThtsContext& ctx) const override;
            virtual std::shared_ptr<ActionVector> get_valid_actions_itfc(
                std::shared_ptr<const State> state,
                ThtsContext& ctx) const override;
            virtual std::shared_ptr<StateDistr> get_transition_distribution_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action,
                ThtsContext& ctx) const override;
            virtual std::shared_ptr<const State> sample_transition_distribution_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action, 
                 RandManager& rand_manager,
                ThtsContext& ctx) const override;
            virtual Eigen::ArrayXd get_mo_reward_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action,
                ThtsContext& ctx) const override;
    };
}
