#pragma once

#include "thts_env.h"
#include "thts_context.h"
#include "thts_manager.h"
#include "thts_types.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

enum FLDirection { FL_RIGHT, FL_DOWN, FL_LEFT, FL_UP };

enum FLRewardType { FL_SPARSE_DISCOUNTED_REWARD, FL_SPARSE_LEN_REWARD, FL_DENSE_REWARD };


// no holes
static const std::string FL_6x6_NO_HOLE_MAP[] = 
{
    "SFFFFF",
    "FFFFFF",
    "FFFFFF",
    "FFFFFF",
    "FFFFFF",
    "FFFFFG",
};

// copied from gymnasium
static const std::string FL_4x4_MAP[] =
{
    "SFFF", 
    "FHFH", 
    "FFFH", 
    "HFFG"
};

// copied from gymnasium
static const std::string FL_8x8_MAP[] =
{
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
};

// python generate_frozen_lake_env.py 5 5 0.8
static const std::string FL_GEN_5x5_MAP[] =
{
    "SFFFF",
    "HFFFH",
    "HHFFF",
    "FFFFF",
    "FFFHG",
};

// python generate_frozen_lake_env.py 6 6 0.8
static const std::string FL_GEN_6x6_MAP[] =
{
    "SFFFFF",
    "FHFHFF",
    "FFFFFF",
    "FFHFFF",
    "FFFFHH",
    "FFFFFG",
};

// python generate_frozen_lake_env.py 4 8 0.8
static const std::string FL_GEN_4x8_MAP[] =
{
    "SFFFHFFF",
    "FHFFFFFF",
    "FHFFHFFF",
    "FFFFFFHG",
};

// python generate_frozen_lake_env.py 4 12 0.8
static const std::string FL_GEN_4x12_MAP[] =
{
    "SFFHFFFFFFFH",
    "FFFFFHFFFFFF",
    "FFFFHFFFFFFF",
    "FFFFFFFFFFHG",
};

// python generate_frozen_lake_env.py 8 12 0.8
static const std::string FL_GEN_8x12_MAP[] =
{
    "SFFFFFFFFHFH",
    "FFFFFFFFFHFF",
    "FFFFFFFFFHHF",
    "FFFFFHFFHFFH",
    "FFFFFFFFFFFF",
    "FFFFFFFFFFFF",
    "FFFFHHFFFFFF",
    "FFHFFFFFFFFG",
};

// python generate_frozen_lake_env.py 12 12 0.8
static const std::string FL_GEN_12x12_MAP[] =
{
    "SFFHFFFFFFFF",
    "FHFFFFFFFHFF",
    "HFFFFHHHFFFF",
    "FFHFFFFFHFFF",
    "FHHHFHHFFFFF",
    "HFHFHFFFHFFF",
    "FFFFFFFFHFFF",
    "HHFFFHFFFFFF",
    "FFFFFFFHFFFH",
    "FFHFFHFFHFFH",
    "FFFFFHHFFFFH",
    "FHFFFFFHFFFG",
};

// python generate_frozen_lake_env.py 8 16 0.8
static const std::string FL_GEN_8x16_MAP[] =
{
    "SFFFFFFFFFFFFFHF",
    "FHHFFFHFHHFFFHFF",
    "FHFFHFFFHFFFFFFF",
    "FFFHFFFHFFFFFHHF",
    "HFFHFHFFFFFFHHFF",
    "FFFHFHFFHHFFFFFF",
    "FFFFFFFFFFFFFFHF",
    "FFFFFHFFHHFFHHFG",
};

// python generate_frozen_lake_env.py 16 16 0.8
static const std::string FL_GEN_16x16_MAP[] =
{
    "SFFHFHFFHFFFFFFF",
    "FFHHFHFFFHFFFHFF",
    "FFFFHFHFHFHHFFFF",
    "FFFHFFFFFHFFFFFF",
    "FFFFHFFFFFFHFHFF",
    "FFFFHFFFHFFFFFFF",
    "FFHHHFHHFHFFHFHH",
    "FFFFFFFFFFFFFFFF",
    "FFFFFFFFFFFFFFHF",
    "HFFFFHHFHFFFHFHF",
    "FFFHFFFFFFFFHFFF",
    "FFFFFFFHFHFFFFFF",
    "FFHHFFHHHHFFFFFF",
    "FFHFFFFHFFFFFFFH",
    "FFFFFFFFFFHFHFFF",
    "FFHFHFHHFFHFFFFG",
};


namespace thts{

    /** 
     * In house implementation of a deterministic frozen lake environment.
     * 
     * In the map representation passed in, S=Start, F=Floor, H=Hole, G=Goal.
     * 
     * Assumes that the map passed is static, constant, will be available for the environment objects lifetime and that 
     * the env doesn't have to clean up the memory.
     * 
     * Member variables:
     *      height: The height of the frozen lake env
     *      width: The width of the frozen lake env
     *      map: An array of strings representing 
     *      cached_actions: Actions are constant, so make array at construction and just return it when want valid acts
     *      reward_discount_factor: Reward returned for reaching goal at time t is (reward_discount_factor)^t
     */
    class FrozenLakeEnv : public ThtsEnv {

        /**
         * Core FrozenLakeEnv implementaion.
         */
        protected:
            int height;
            int width;
            const std::string* map;
            int reward_type;
            double reward_discount_factor;
            int max_steps;
            bool is_slippery;
            double dense_hole_cost;
            bool avoid_collision_actions;


        /**
         * Core ThtsEnv implementation functinos.
         */
        public:
            /**
             * Constructor
             */
            FrozenLakeEnv(
                int width, 
                int height, 
                const std::string* map, 
                bool is_slippery=false,
                int reward_type=FL_DENSE_REWARD, 
                double reward_discount_factor=0.99, 
                int max_steps=-1,
                double dense_hole_cost=100.0,
                bool avoid_collision_actions=false);

            virtual std::shared_ptr<ThtsEnv> clone() override;

            /**
             * Mark destructor as virtual for subclassing.
             */
            virtual ~FrozenLakeEnv() = default;

            /**
             * Returns the initial state for the environment.
             * 
             * Returns:
             *      Initial state for this environment instance
             */
            std::shared_ptr<const Int3TupleState> get_initial_state() const;

            /**
             * Returns if a state is a sink state.
             * 
             * Args:
             *      state: The state to be checked if it is a sink state
             * 
             * Returns:
             *      True if 'state' is a sink state and false otherwise
             */
            bool is_sink_state(std::shared_ptr<const Int3TupleState> state) const;

            /**
             * Returns a list of actions that are valid in a given state.
             * 
             * Args:
             *      state: The state that we want a list of available actions from
             * 
             * Returns:
             *      Returns a list of actions available from 'state'
             */
            std::shared_ptr<IntActionVector> get_valid_actions(std::shared_ptr<const Int3TupleState> state) const;

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
            std::shared_ptr<Int3TupleStateDistr> get_transition_distribution(
                std::shared_ptr<const Int3TupleState> state, std::shared_ptr<const IntAction> action) const;

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
            std::shared_ptr<const Int3TupleState> sample_transition_distribution(
                std::shared_ptr<const Int3TupleState> state, 
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
            double get_reward(
                std::shared_ptr<const Int3TupleState> state, 
                std::shared_ptr<const IntAction> action) const;



        /**
         * Boilerplate functinos (defined in thts_env_template.{h,cpp}) using the default implementations provided by 
         * thts_env.{h,cpp}. 
         */
        public:
            /**
             * Returns a distribution over observations from a (next) state, action pair.
             * 
             * Given a state and action returns a distribution of possible observations. The probability 
             * distribution is returned in the form of a map, where the keys are of the Observation type, and the 
             * values are doubles, which sum to one.
             * 
             * A default implementation is provided for full observable environments, where observation == next state.
             * 
             * Args:
             *      action: The action to get an observation distribution for
             *      next_state: The state (arriving in)  to get an observation distribution from
             * 
             * Returns:
             *      Returns a distribution over observations from taking 'action' in state 'state'.
             */
            virtual std::shared_ptr<Int3TupleStateDistr> get_observation_distribution(
                std::shared_ptr<const IntAction> action, std::shared_ptr<const Int3TupleState> next_state, ThtsContext& ctx) const;

            /**
             * Samples an observation when arriving in a (next) state after taking an action.
             * 
             * Given a state-action pair, samples a possible sobservation.
             * 
             * A default implementation is provided for full observable environments, where observation == next state.
             * 
             * Args:
             *      action: The action taken to sample an observation for
             *      next_state: The state (arriving in)  to sample an observation for
             *      rand_manager: A RandManager ref to access the random number sampling interface
             * 
             * Returns:
             *      Returns an observation sampled from taking 'action' that arived in 'next_state'
             */
            virtual std::shared_ptr<const Int3TupleState> sample_observation_distribution(
                std::shared_ptr<const IntAction> action, 
                std::shared_ptr<const Int3TupleState> next_state, 
                RandManager& rand_manager, ThtsContext& ctx) const;

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
             *      A ThtsContext object, that will be passed to the Thts functions for a single trial, used to 
             *      provide some context or space for caching.
             */
            virtual std::shared_ptr<ThtsContext> sample_context(int tid, RandManager& rand_manager) const;



        /**
         * ThtsEnv interface function definitions. Boilerplate implementations provided from thts_env_template.h
         */
        public:
            virtual std::shared_ptr<const State> get_initial_state_itfc() const override;
            virtual bool is_sink_state_itfc(std::shared_ptr<const State> state, ThtsContext& ctx) const override;
            virtual std::shared_ptr<ActionVector> get_valid_actions_itfc(std::shared_ptr<const State> state, ThtsContext& ctx) const override;
            virtual std::shared_ptr<StateDistr> get_transition_distribution_itfc(
                std::shared_ptr<const State> state, std::shared_ptr<const Action> action, ThtsContext& ctx) const override;
            virtual std::shared_ptr<const State> sample_transition_distribution_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action, 
                 RandManager& rand_manager, ThtsContext& ctx) const override;
            virtual std::shared_ptr<ObservationDistr> get_observation_distribution_itfc(
                std::shared_ptr<const Action> action, std::shared_ptr<const State> next_state, ThtsContext& ctx) const override;
            virtual std::shared_ptr<const Observation> sample_observation_distribution_itfc(
                std::shared_ptr<const Action> action, 
                std::shared_ptr<const State> next_state, 
                 RandManager& rand_manager, ThtsContext& ctx) const override;
            virtual double get_reward_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action, 
                ThtsContext& ctx) const override;
            virtual std::shared_ptr<ThtsContext> sample_context_itfc(int tid, RandManager& rand_manager) const override;
        
        /**
         * Implemented in thts_env.{h,cpp}
         */
        // public:
        //     bool is_fully_observable();
    };
}

