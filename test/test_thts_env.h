#pragma once

#include "thts_env.h"
#include "thts_manager.h"
#include "thts_types.h"

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
     * TODO: docstring
     */
    void run_thts_env_tests();

    /**
     * TODO: docstring
     * 
     * TODO: add tests for probabilistic outcomes at some point
     */  
    class TestThtsManager : public ThtsManager {
        private:
            size_t int_indx;
            vector<int> int_mock_numbers;
            size_t uniform_indx;
            vector<double> uniform_mock_numbers;

        public:          
            TestThtsManager(
                bool mcts_mode=true, 
                bool use_transposition_table=false, 
                bool is_two_player_game=false,
                HeuristicFnPtr heuristic_fn_ptr=helper::zero_heuristic_fn,
                PriorFnPtr prior_fn_ptr=nullptr,
                int seed=60415) :
                    ThtsManager(
                        mcts_mode, 
                        use_transposition_table,
                        is_two_player_game,
                        heuristic_fn_ptr,
                        prior_fn_ptr,
                        seed),
                    int_indx(0),
                    uniform_indx(0) {};

            void push_random_ints(vector<int>& rand_ints) {
                int_mock_numbers.insert(int_mock_numbers.end(), rand_ints.begin(), rand_ints.end());
            }

            void push_random_uniforms(vector<double>& rand_uniforms) {
                uniform_mock_numbers.insert(uniform_mock_numbers.end(), rand_uniforms.begin(), rand_uniforms.end());
            }

            virtual int get_rand_int(int min_included, int max_excluded) {
                if (int_indx < int_mock_numbers.size()) {
                    return int_mock_numbers[int_indx++];
                }
                return ThtsManager::get_rand_int(min_included,max_excluded);
            };

            virtual double get_rand_uniform() {
                if (uniform_indx < uniform_mock_numbers.size()) {
                    return uniform_mock_numbers[uniform_indx++];
                }
                return ThtsManager::get_rand_uniform();
            };
    };

    /** 
     * Implement a simple ThtsEnv subclass to be used for testing
     * TODO: document class properly
     * TODO:
     */
    class TestThtsEnv : public ThtsEnv {

        private:
            int grid_size;
            double stay_prob;

        protected:
            string env_id = "test_thts_env_id";

        public:
            /**
             * Node implementation
             */
            TestThtsEnv(int grid_size, double stay_prob=0.0);

            shared_ptr<const IntPairState> get_initial_state() const;

            bool is_sink_state(shared_ptr<const IntPairState> state) const;

            shared_ptr<StringActionVector> get_valid_actions(shared_ptr<const IntPairState> state) const;

            shared_ptr<IntPairStateDistr> get_transition_distribution(
                shared_ptr<const IntPairState> state, shared_ptr<const StringAction> action) const;

            shared_ptr<const IntPairState> sample_transition_distribution(
                shared_ptr<const IntPairState> state, 
                shared_ptr<const StringAction> action, 
                shared_ptr<TestThtsManager> manager) const;

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
                shared_ptr<const State> state, shared_ptr<const Action> action, shared_ptr<ThtsManager> manager) const 
            {
                shared_ptr<const IntPairState> state_itfc = static_pointer_cast<const IntPairState>(state);
                shared_ptr<const StringAction> action_itfc = static_pointer_cast<const StringAction>(action);
                shared_ptr<TestThtsManager> manager_itfc = static_pointer_cast<TestThtsManager>(manager);
                shared_ptr<const IntPairState> obsv = sample_transition_distribution(
                    state_itfc, action_itfc, manager_itfc);
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