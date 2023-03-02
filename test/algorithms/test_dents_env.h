#pragma once

#include "thts_env.h"
#include "thts_manager.h"
#include "thts_types.h"

#include "test/test_thts_manager.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <iostream>

namespace thts_test{
    using namespace std;
    using namespace thts;


    /** 
     * A ThtsEnv to test Dents ability to identify and start sampling an optimal path.
     * 
     * There are two paths, the 'gud' path, and the 'bad' path. On the bad path, a reward of 'bad_reward' is recieved 
     * on any action (so entropy will be high), on the 'gud' path, only taking the action 0 will give a reward of 
     * 'gud_reward' and any other action gives a reward of 0 (so low entropy).
     * 
     * Implementation overview:
     * States are of the form of an integer pair. The first will be 0 or 1, which denotes if we're on the gud/bad path 
     * respectively. The second is how far along the chain we are. The initial state is (-1,-1), to denote that a path 
     * hasn't been chosen yet. From the initial state there are two actions 0 and 1, which lead to (0,0) and (1,0) 
     * respectively. After that, any action from 0, ..., 'num_actions' can be chosen, and always results in (x,y) 
     * being updated to (x,y+1).
     */
    class TestDentsThtsEnv : public ThtsEnv {

        private:
            int chain_length;
            int num_actions;
            double gud_reward;
            double bad_reward;

        public:
            TestDentsThtsEnv(int chain_length=4, int num_actions=5, double gud_reward=1.0, double bad_reward=0.1) : 
                ThtsEnv(true), 
                chain_length(chain_length), 
                num_actions(num_actions), 
                gud_reward(gud_reward), 
                bad_reward(bad_reward) 
            {
            }

            virtual ~TestDentsThtsEnv() = default;

            shared_ptr<const IntPairState> get_initial_state() const {
                return make_shared<IntPairState>(IntPairState(-1,-1));
            }

            bool is_sink_state(shared_ptr<const IntPairState> state) const {
                return state->state.second == chain_length;
            }

            shared_ptr<IntActionVector> get_valid_actions(shared_ptr<const IntPairState> state) const {
                shared_ptr<IntActionVector> valid_actions = make_shared<IntActionVector>();

                if (is_sink_state(state)) {
                    return valid_actions;
                }
                
                valid_actions->push_back(make_shared<const IntAction>(0));
                valid_actions->push_back(make_shared<const IntAction>(1));

                if (state->state.first > -1) {
                    for (int i=2; i < num_actions; i++) {
                        valid_actions->push_back(make_shared<const IntAction>(i));
                    }
                }

                return valid_actions;
            }

            shared_ptr<IntPairStateDistr> get_transition_distribution(
                shared_ptr<const IntPairState> state, shared_ptr<const IntAction> action) const 
            {
                shared_ptr<const IntPairState> next_state = sample_transition_distribution(state,action);
                shared_ptr<IntPairStateDistr> transition_distribution = make_shared<IntPairStateDistr>(); 
                transition_distribution->insert_or_assign(next_state, 1.0);
                return transition_distribution;
            }

            shared_ptr<const IntPairState> sample_transition_distribution(
                shared_ptr<const IntPairState> state, 
                shared_ptr<const IntAction> action) const 
            {
                shared_ptr<IntPairState> new_state = make_shared<IntPairState>(0,0);
                if (state->state.first == -1) {
                    new_state->state.first = action->action;
                } else {
                    new_state->state.first = state->state.first;
                }
                new_state->state.second = state->state.second + 1;

                return new_state;
            }

            shared_ptr<const IntPairState> sample_transition_distribution(
                shared_ptr<const IntPairState> state, 
                shared_ptr<const IntAction> action, 
                RandManager& rand_manager) const 
            {
                return sample_transition_distribution(state,action);
            }

            double get_reward(
                shared_ptr<const IntPairState> state, 
                shared_ptr<const IntAction> action, 
                shared_ptr<const IntPairState> observation=nullptr) const 
            {
                if (state->state.first == -1) {
                    return 0.0;
                } 

                if (state->state.first == 0) {
                    if (action->action == 0) {
                        return gud_reward;
                    }
                    return 0.0;
                }

                // if (state->state.first == 1)
                return bad_reward;
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
                shared_ptr<const State> state, shared_ptr<const Action> action, RandManager& rand_manager) const 
            {
                shared_ptr<const IntPairState> state_itfc = static_pointer_cast<const IntPairState>(state);
                shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
                shared_ptr<const IntPairState> next_state = sample_transition_distribution(
                    state_itfc, action_itfc, rand_manager);
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
                RandManager& rand_manager) const 
            {
                return thts::ThtsEnv::sample_observation_distribution_itfc(action, next_state, rand_manager);
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