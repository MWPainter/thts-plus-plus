#pragma once

#include "thts_env.h"
#include "thts_manager.h"
#include "thts_types.h"

#include "test_thts_manager.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace thts_test{
    using namespace std;
    using namespace thts;


    /** 
     * A ThtsEnv that will be used throughout testing. We use these tests to test this class and the interface.
     * 
     * This class also served as the original implementation that the template was based off.
     * 
     * The environment is a grid, where we are not allowed to move out of bounds, start in one corner, and want to get 
     * to the opposing corner.
     * 
     * Member variables:
     *      grid_size: The size of the environment ((grid_size+1) x (grid_size+1) grid)
     *      stay_prob: The probability that an action results in not moving at all (so can test stochastic envs too)
     */
    class TestThtsEnv : public ThtsEnv {

        private:
            int grid_size;
            double stay_prob;

        /**
         * Node implementation
         */
        public:
            TestThtsEnv(int grid_size, double stay_prob=0.0) : 
                ThtsEnv(true), grid_size(grid_size), stay_prob(stay_prob) {}

            virtual ~TestThtsEnv() = default;

            shared_ptr<const IntPairState> get_initial_state() const {
                return make_shared<IntPairState>(IntPairState(0,0));
            }

            bool is_sink_state(shared_ptr<const IntPairState> state) const {
                return state->state == make_pair(grid_size, grid_size);
            }

            shared_ptr<StringActionVector> get_valid_actions(shared_ptr<const IntPairState> state) const {
                shared_ptr<StringActionVector> valid_actions = make_shared<StringActionVector>();

                if (is_sink_state(state)) {
                    return valid_actions;
                }

                int x = state->state.first;
                if (x > 0) {
                    valid_actions->push_back(make_shared<const StringAction>("left"));
                } 
                if (x < grid_size) {
                    valid_actions->push_back(make_shared<const StringAction>("right"));
                }

                int y = state->state.second;
                if (y > 0) {
                    valid_actions->push_back(make_shared<const StringAction>("up"));
                } 
                if (y < grid_size) {
                    valid_actions->push_back(make_shared<const StringAction>("down"));
                }

                return valid_actions;
            }

        private:
            shared_ptr<const IntPairState> make_candidate_next_state(
                shared_ptr<const IntPairState> state, shared_ptr<const StringAction> action) const
            {
                shared_ptr<IntPairState> new_state = make_shared<IntPairState>(state->state);
                if (action->action == "left") {
                    new_state->state.first -= 1;
                } else if (action->action == "right") {
                    new_state->state.first += 1;
                } else if (action->action == "up") {
                    new_state->state.second -= 1;
                } else if (action->action == "down") {
                    new_state->state.second += 1;
                }
                return new_state;
            }

        public:
            shared_ptr<IntPairStateDistr> get_transition_distribution(
                shared_ptr<const IntPairState> state, shared_ptr<const StringAction> action) const 
            {
                shared_ptr<const IntPairState> new_state = make_candidate_next_state(state, action);
                shared_ptr<IntPairStateDistr> transition_distribution = make_shared<IntPairStateDistr>(); 
                transition_distribution->insert_or_assign(new_state, 1.0-stay_prob);
                if (stay_prob > 0.0) {
                    transition_distribution->insert_or_assign(state, stay_prob);
                }
                return transition_distribution;
            }

            shared_ptr<const IntPairState> sample_transition_distribution(
                shared_ptr<const IntPairState> state, 
                shared_ptr<const StringAction> action, 
                shared_ptr<ThtsManager> manager) const 
            {
                if (stay_prob > 0.0) {
                    double sample = manager->get_rand_uniform();
                    if (sample < stay_prob) {
                        return state;
                    }
                }

                return make_candidate_next_state(state,action);
            }

            double get_reward(
                shared_ptr<const IntPairState> state, 
                shared_ptr<const StringAction> action, 
                shared_ptr<const IntPairState> observation=nullptr) const 
            {
                return -1.0;
            }

        /**
         * Interface implementation (basically calls the above implementations with surrounding casts).
         */
        public:
            virtual shared_ptr<const State> get_initial_state_itfc() const {
                shared_ptr<const IntPairState> init_state = get_initial_state();
                return static_pointer_cast<const State>(init_state);
            }

            virtual bool is_sink_state_itfc(shared_ptr<const State> state) const {
                shared_ptr<const IntPairState> state_itfc = static_pointer_cast<const IntPairState>(state);
                return is_sink_state(state_itfc);
            }

            virtual shared_ptr<ActionVector> get_valid_actions_itfc(shared_ptr<const State> state) const {
                shared_ptr<const IntPairState> state_itfc = static_pointer_cast<const IntPairState>(state);
                shared_ptr<StringActionVector> valid_actions_itfc = get_valid_actions(state_itfc);

                shared_ptr<ActionVector> valid_actions = make_shared<ActionVector>();
                for (shared_ptr<const StringAction> act : *valid_actions_itfc) {
                    valid_actions->push_back(static_pointer_cast<const Action>(act));
                }
                return valid_actions;
            }

            virtual shared_ptr<StateDistr> get_transition_distribution_itfc(
                shared_ptr<const State> state, shared_ptr<const Action> action) const 
            {
                shared_ptr<const IntPairState> state_itfc = static_pointer_cast<const IntPairState>(state);
                shared_ptr<const StringAction> action_itfc = static_pointer_cast<const StringAction>(action);
                shared_ptr<IntPairStateDistr> distr_itfc = get_transition_distribution(state_itfc, action_itfc);
                
                shared_ptr<StateDistr> distr = make_shared<StateDistr>(); 
                for (pair<shared_ptr<const IntPairState>,double> key_val_pair : *distr_itfc) {
                    shared_ptr<const State> state = static_pointer_cast<const State>(key_val_pair.first);
                    double prob = key_val_pair.second;
                    distr->insert_or_assign(state, prob);
                }
                return distr;
            }

            virtual shared_ptr<const State> sample_transition_distribution_itfc(
                shared_ptr<const State> state, shared_ptr<const Action> action, shared_ptr<ThtsManager> manager) const 
            {
                shared_ptr<const IntPairState> state_itfc = static_pointer_cast<const IntPairState>(state);
                shared_ptr<const StringAction> action_itfc = static_pointer_cast<const StringAction>(action);
                shared_ptr<ThtsManager> manager_itfc = static_pointer_cast<ThtsManager>(manager);
                shared_ptr<const IntPairState> next_state = sample_transition_distribution(
                    state_itfc, action_itfc, manager_itfc);
                return static_pointer_cast<const State>(next_state);
            }

            virtual std::shared_ptr<ObservationDistr> get_observation_distribution_itfc(
                std::shared_ptr<const Action> action, std::shared_ptr<const State> next_state) const
            {
                return thts::ThtsEnv::get_observation_distribution_itfc(action, next_state);
            }

            virtual std::shared_ptr<const Observation> sample_observation_distribution_itfc(
                std::shared_ptr<const Action> action, 
                std::shared_ptr<const State> next_state, 
                std::shared_ptr<ThtsManager> thts_manager) const 
            {
                return thts::ThtsEnv::sample_observation_distribution_itfc(action, next_state, thts_manager);
            }

            virtual double get_reward_itfc(
                shared_ptr<const State> state, 
                shared_ptr<const Action> action, 
                shared_ptr<const Observation> observation=nullptr) const
            {
                shared_ptr<const IntPairState> state_itfc = static_pointer_cast<const IntPairState>(state);
                shared_ptr<const StringAction> action_itfc = static_pointer_cast<const StringAction>(action);
                shared_ptr<const IntPairState> obsv_itfc = static_pointer_cast<const IntPairState>(observation);
                return get_reward(state_itfc, action_itfc, obsv_itfc);
            }
    };



    /** 
     * A test env that is used to test two player games. 
     * 
     * If game is of length n, then at the ith (0<=i<n) step, the player can chose between a reward of 2^(n-1-i) or 
     * -2^(n-1-i).
     * 
     * States take the form of (game_step, cumulative_score)
     * 
     * Member variables:
     *      game_len: Length of the game
     *      consistent_actions: If we want the actions to always be "1","-1". (So we can test Rents)
     */
    class TestThtsGameEnv : public ThtsEnv {

        private:
            int game_len;
            bool consistent_actions;

        /**
         * Node implementation
         */
        public:
            TestThtsGameEnv(int game_len, bool consistent_actions=false) : 
                ThtsEnv(true), game_len(game_len), consistent_actions(consistent_actions) {}

            virtual ~TestThtsGameEnv() = default;

            shared_ptr<const IntPairState> get_initial_state() const {
                return make_shared<IntPairState>(IntPairState(0,0));
            }

            bool is_sink_state(shared_ptr<const IntPairState> state) const {
                return state->state.first == game_len;
            }

            shared_ptr<IntActionVector> get_valid_actions(shared_ptr<const IntPairState> state) const {
                int game_step = state->state.first;
                int rew = 1 << (game_len - 1 - game_step);
                if (consistent_actions) rew = 1;

                shared_ptr<IntActionVector> valid_actions = make_shared<IntActionVector>();
                if (!is_sink_state(state)) {
                    valid_actions->push_back(make_shared<const IntAction>(rew));
                    valid_actions->push_back(make_shared<const IntAction>(-rew));
                }

                return valid_actions;
            }

        public:
            shared_ptr<IntPairStateDistr> get_transition_distribution(
                shared_ptr<const IntPairState> state, shared_ptr<const IntAction> action) const 
            {
                shared_ptr<const IntPairState> new_state = sample_transition_distribution(state,action);
                shared_ptr<IntPairStateDistr> transition_distribution = make_shared<IntPairStateDistr>(); 
                transition_distribution->insert_or_assign(new_state, 1.0);
                return transition_distribution;
            }

            shared_ptr<const IntPairState> sample_transition_distribution(
                shared_ptr<const IntPairState> state, 
                shared_ptr<const IntAction> action, 
                shared_ptr<ThtsManager> manager=nullptr) const 
            {
                int last_game_step = state->state.first;
                int cumulative_score = state->state.second;
                int step_score = action->action;
                if (consistent_actions) step_score *= 1 << (game_len - 1 - last_game_step);
                return make_shared<const IntPairState>(last_game_step+1, cumulative_score+step_score);
            }

            double get_reward(
                shared_ptr<const IntPairState> state, 
                shared_ptr<const IntAction> action, 
                shared_ptr<const IntPairState> observation=nullptr) const 
            {
                int game_step = state->state.first;
                if (game_step < game_len-1) {
                    return 0;
                }

                int cumulative_score = state->state.second;
                int step_score = action->action;
                if (consistent_actions) step_score *= 1 << (game_len - 1 - game_step);
                return cumulative_score + step_score;

            }

        /**
         * Interface implementation (basically calls the above implementations with surrounding casts).
         */
        public:
            virtual shared_ptr<const State> get_initial_state_itfc() const {
                shared_ptr<const IntPairState> init_state = get_initial_state();
                return static_pointer_cast<const State>(init_state);
            }

            virtual bool is_sink_state_itfc(shared_ptr<const State> state) const {
                shared_ptr<const IntPairState> state_itfc = static_pointer_cast<const IntPairState>(state);
                return is_sink_state(state_itfc);
            }

            virtual shared_ptr<ActionVector> get_valid_actions_itfc(shared_ptr<const State> state) const {
                shared_ptr<const IntPairState> state_itfc = static_pointer_cast<const IntPairState>(state);
                shared_ptr<IntActionVector> valid_actions_itfc = get_valid_actions(state_itfc);

                shared_ptr<ActionVector> valid_actions = make_shared<ActionVector>();
                for (shared_ptr<const IntAction> act : *valid_actions_itfc) {
                    valid_actions->push_back(static_pointer_cast<const Action>(act));
                }
                return valid_actions;
            }

            virtual shared_ptr<StateDistr> get_transition_distribution_itfc(
                shared_ptr<const State> state, shared_ptr<const Action> action) const 
            {
                shared_ptr<const IntPairState> state_itfc = static_pointer_cast<const IntPairState>(state);
                shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
                shared_ptr<IntPairStateDistr> distr_itfc = get_transition_distribution(state_itfc, action_itfc);
                
                shared_ptr<StateDistr> distr = make_shared<StateDistr>(); 
                for (pair<shared_ptr<const IntPairState>,double> key_val_pair : *distr_itfc) {
                    shared_ptr<const State> state = static_pointer_cast<const State>(key_val_pair.first);
                    double prob = key_val_pair.second;
                    distr->insert_or_assign(state, prob);
                }
                return distr;
            }

            virtual shared_ptr<const State> sample_transition_distribution_itfc(
                shared_ptr<const State> state, shared_ptr<const Action> action, shared_ptr<ThtsManager> manager) const 
            {
                shared_ptr<const IntPairState> state_itfc = static_pointer_cast<const IntPairState>(state);
                shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
                shared_ptr<ThtsManager> manager_itfc = static_pointer_cast<ThtsManager>(manager);
                shared_ptr<const IntPairState> next_state = sample_transition_distribution(
                    state_itfc, action_itfc, manager_itfc);
                return static_pointer_cast<const State>(next_state);
            }

            virtual std::shared_ptr<ObservationDistr> get_observation_distribution_itfc(
                std::shared_ptr<const Action> action, std::shared_ptr<const State> next_state) const
            {
                return thts::ThtsEnv::get_observation_distribution_itfc(action, next_state);
            }

            virtual std::shared_ptr<const Observation> sample_observation_distribution_itfc(
                std::shared_ptr<const Action> action, 
                std::shared_ptr<const State> next_state, 
                std::shared_ptr<ThtsManager> thts_manager) const 
            {
                return thts::ThtsEnv::sample_observation_distribution_itfc(action, next_state, thts_manager);
            }

            virtual double get_reward_itfc(
                shared_ptr<const State> state, 
                shared_ptr<const Action> action, 
                shared_ptr<const Observation> observation=nullptr) const
            {
                shared_ptr<const IntPairState> state_itfc = static_pointer_cast<const IntPairState>(state);
                shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
                shared_ptr<const IntPairState> obsv_itfc = static_pointer_cast<const IntPairState>(observation);
                return get_reward(state_itfc, action_itfc, obsv_itfc);
            }
    };
}