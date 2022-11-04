#pragma once

#include "thts_env.h"
#include "thts_types.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace thts_test{
    using namespace std;
    using namespace thts;
    
    /**
     * TODO: docstring
     */
    void run_thts_env_tests();

    /** 
     * Implement a simple ThtsEnv subclass to be used for testing
     * TODO: document class properly
     * TODO:
     */
    class TestThtsEnv : public ThtsEnv {

        private:
            int grid_size;

        protected:
            string env_id = "test_thts_env_id";

        public:
            /**
             * Node implementation
             */
            TestThtsEnv(int grid_size);

            shared_ptr<const IntPairState> get_initial_state() const;

            bool is_sink_state(shared_ptr<const IntPairState> state) const;

            shared_ptr<StringActionVector> get_valid_actions(shared_ptr<const IntPairState> state) const;

            shared_ptr<IntPairStateDistr> get_transition_distribution(
                shared_ptr<const IntPairState> state, shared_ptr<const StringAction> action) const;

            shared_ptr<const IntPairState> sample_transition_distribution(
                shared_ptr<const IntPairState> state, shared_ptr<const StringAction> action) const;

            double get_reward(
                shared_ptr<const IntPairState> state, 
                shared_ptr<const StringAction> action, 
                shared_ptr<const IntPairState> observation=nullptr) const;

            /**
             * Interface implementation (basically calls the above implementations with surrounding casts).
             */
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

            virtual shared_ptr<ObservationDistr> get_transition_distribution_itfc(
                shared_ptr<const State> state, shared_ptr<const Action> action) const 
            {
                shared_ptr<const IntPairState> state_itfc = static_pointer_cast<const IntPairState>(state);
                shared_ptr<const StringAction> action_itfc = static_pointer_cast<const StringAction>(action);
                shared_ptr<IntPairStateDistr> distr_itfc = get_transition_distribution(state_itfc, action_itfc);
                
                shared_ptr<ObservationDistr> distr = make_shared<ObservationDistr>(); 
                for (pair<shared_ptr<const IntPairState>,double> key_val_pair : *distr_itfc) {
                    shared_ptr<const Observation> obsv = static_pointer_cast<const Observation>(key_val_pair.first);
                    double prob = key_val_pair.second;
                    distr->insert_or_assign(obsv, prob);
                }
                return distr;
            }

            virtual shared_ptr<const Observation> sample_transition_distribution_itfc(
                shared_ptr<const State> state, shared_ptr<const Action> action) const 
            {
                shared_ptr<const IntPairState> state_itfc = static_pointer_cast<const IntPairState>(state);
                shared_ptr<const StringAction> action_itfc = static_pointer_cast<const StringAction>(action);
                shared_ptr<const IntPairState> obsv = sample_transition_distribution(state_itfc, action_itfc);
                return static_pointer_cast<const Observation>(obsv);
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
}