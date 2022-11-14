#include "test/test_thts_env.h"

#include <iostream>
#include <sstream>

using namespace std;
using namespace thts;
using namespace thts_test;

namespace thts_test{
    // /** 
    //  * TODO: comment properly
    //  */
    // typedef pair<int,int> State;

    /** 
     * Implement a simple ThtsEnv subclass to be used for testing
     * TODO: document class properly
     */ 
    TestThtsEnv::TestThtsEnv(int grid_size, double stay_prob) : grid_size(grid_size), stay_prob(stay_prob) {}

    shared_ptr<const IntPairState> TestThtsEnv::get_initial_state() const {
        return make_shared<IntPairState>(IntPairState(0,0));
    }

    bool TestThtsEnv::is_sink_state(shared_ptr<const IntPairState> state) const {
        return state->state == make_pair(grid_size, grid_size);
    }

    shared_ptr<StringActionVector> TestThtsEnv::get_valid_actions(shared_ptr<const IntPairState> state) const {
        shared_ptr<StringActionVector> valid_actions = make_shared<StringActionVector>();

        if (is_sink_state(state)) {
            return valid_actions;
        }

        int x = state->state.first;
        if (x > 0) {
            valid_actions->push_back(make_shared<const StringAction>("left"));
        } else if (x < grid_size) {
            valid_actions->push_back(make_shared<const StringAction>("right"));
        }

        int y = state->state.second;
        if (y > 0) {
            valid_actions->push_back(make_shared<const StringAction>("up"));
        } else if (y < grid_size) {
            valid_actions->push_back(make_shared<const StringAction>("down"));
        }

        return valid_actions;
    }

    shared_ptr<IntPairState> make_candidate_next_state(
        shared_ptr<const IntPairState> state, shared_ptr<const StringAction> action) 
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

    shared_ptr<IntPairStateDistr> TestThtsEnv::get_transition_distribution(
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

    shared_ptr<const IntPairState> TestThtsEnv::sample_transition_distribution(
        shared_ptr<const IntPairState> state, 
        shared_ptr<const StringAction> action, 
        shared_ptr<TestThtsManager> manager) const 
    {
        if (stay_prob > 0.0) {
            double sample = manager->get_rand_uniform();
            if (sample < stay_prob) {
                return state;
            }
        }

        return make_candidate_next_state(state,action);
    }

    double TestThtsEnv::get_reward(
        shared_ptr<const IntPairState> state, 
        shared_ptr<const StringAction> action, 
        shared_ptr<const IntPairState> Observation) const 
    {
        return -1.0;
    }



    

    /**
     * TODO: document
     */
    void run_thts_env_tests() {
        cout << "----------" << endl;
        cout << "Testing ThtsEnv using a test subclass." << endl;
        cout << "----------" << endl;
        
        // Test all the functions of the test thts env by iterating through a few states with fixed actions
        TestThtsEnv env(1);
        shared_ptr<TestThtsManager> manager = make_shared<TestThtsManager>();

        shared_ptr<const IntPairState> init_state = env.get_initial_state();

        cout << *(init_state) << endl;
        cout << *(env.get_valid_actions(init_state)) << endl;
        cout << env.is_sink_state(init_state) << endl << endl;

        shared_ptr<const StringAction> act = make_shared<const StringAction>("right");
        shared_ptr<const IntPairState> next_state = env.sample_transition_distribution(init_state, act, manager);

        cout << env.get_reward(init_state, act) << endl;
        cout << *(next_state) << endl;
        cout << *(env.get_valid_actions(next_state)) << endl;
        cout << env.is_sink_state(next_state) << endl << endl;

        act = make_shared<const StringAction>("down");
        next_state = env.sample_transition_distribution(next_state, act, manager);

        cout << env.get_reward(init_state, act) << endl;
        cout << *(next_state) << endl;
        cout << *(env.get_valid_actions(next_state)) << endl;
        cout << env.is_sink_state(next_state) << endl << endl;

        // Also check that getting the entirer transition distribution happens correctly
        shared_ptr<IntPairStateDistr> distribution = env.get_transition_distribution(init_state, act);
        cout << *(distribution) << endl << endl;
        cout << *(TestThtsEnv(2,0.5).get_transition_distribution(init_state,act)) << endl << endl;

        // Repeat all of the above, but using the interface, AND, only printing out at the end to check shared memory 
        // is allocated correctly. (Should get some garbage if not).
        shared_ptr<const State> state_0 = env.get_initial_state_itfc();
        shared_ptr<ActionVector> actions_0 = env.get_valid_actions_itfc(state_0);
        shared_ptr<const StringAction> act_string_0 = make_shared<const StringAction>("right");

        shared_ptr<const Action> act_0 = static_pointer_cast<const Action>(act_string_0);
        shared_ptr<const Observation> obsv_1 = env.sample_transition_distribution_itfc(state_0, act_0, manager);
        shared_ptr<const State> state_1 = static_pointer_cast<const State>(obsv_1);
        shared_ptr<ActionVector> actions_1 = env.get_valid_actions_itfc(state_1);
        shared_ptr<const StringAction> act_string_1 = make_shared<const StringAction>("down");
        shared_ptr<const Action> act_1 = static_pointer_cast<const Action>(act_string_1);

        shared_ptr<const Observation> obsv_2 = env.sample_transition_distribution_itfc(state_1, act_1, manager);
        shared_ptr<const State> state_2 = static_pointer_cast<const State>(obsv_2);
        shared_ptr<ActionVector> actions_2 = env.get_valid_actions_itfc(state_2);


        cout << *(state_0) << endl;
        cout << *(actions_0) << endl;
        cout << env.is_sink_state_itfc(state_0) << endl << endl;

        cout << env.get_reward_itfc(state_0, act_0) << endl;
        cout << *(state_1) << endl;
        cout << *(actions_1) << endl;
        cout << env.is_sink_state_itfc(state_1) << endl << endl;

        cout << env.get_reward_itfc(state_1, act_1) << endl;
        cout << *(state_2) << endl;
        cout << *(actions_2) << endl;
        cout << env.is_sink_state_itfc(state_2) << endl << endl;
        
        shared_ptr<ObservationDistr> distr_0 = env.get_transition_distribution_itfc(state_0, act_0);
        cout << *(distr_0) << endl << endl;
        cout << *(TestThtsEnv(2,0.5).get_transition_distribution_itfc(init_state,act)) << endl << endl;


    }
}