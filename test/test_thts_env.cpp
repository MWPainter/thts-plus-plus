#include "test_thts_env.h"

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
    TestThtsEnv::TestThtsEnv(int grid_size) : grid_size(grid_size) {}

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
            valid_actions->push_back(make_shared<const StringAction>(StringAction("left")));
        } else if (x < grid_size) {
            valid_actions->push_back(make_shared<const StringAction>(StringAction("right")));
        }

        int y = state->state.second;
        if (y > 0) {
            valid_actions->push_back(make_shared<const StringAction>(StringAction("up")));
        } else if (y < grid_size) {
            valid_actions->push_back(make_shared<const StringAction>(StringAction("down")));
        }

        return valid_actions;
    }

    shared_ptr<IntPairStateDistr> TestThtsEnv::get_transition_distribution(
        shared_ptr<const IntPairState> state, shared_ptr<const StringAction> action) const 
    {
        shared_ptr<const IntPairState> new_state = sample_transition_distribution(state, action);
        shared_ptr<IntPairStateDistr> transition_distribution = make_shared<IntPairStateDistr>(); 
        transition_distribution->insert_or_assign(new_state, 1.0);
        return transition_distribution;
    }

    shared_ptr<const IntPairState> TestThtsEnv::sample_transition_distribution(
        shared_ptr<const IntPairState> state, shared_ptr<const StringAction> action) const 
    {
        shared_ptr<IntPairState> new_state = make_shared<IntPairState>(IntPairState(state->state));
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

    double TestThtsEnv::get_reward(
        shared_ptr<const IntPairState> state, 
        shared_ptr<const StringAction> action, 
        shared_ptr<const IntPairState> Observation) const 
    {
        return -1.0;
    }
    


    // /**
    //  * Helper to print out test states.
    //  */
    // string get_state_string(shared_ptr<const IntPairState> state) {
    //     stringstream ss;
    //     ss << "(" << state->state.first << "," << state->state.second << ")";
    //     return ss.str();
    // }

    // string get_state_string(shared_ptr<const State> state) {
    //     shared_ptr<const IntPairState> cast_state = static_pointer_cast<const IntPairState>(state);
    //     return get_state_string(cast_state);
    // }

    // /**
    //  * Helper to print out lists of valid actions.
    //  */
    // string get_vector_string(shared_ptr<StringActionVector> vec) {
    //     stringstream ss;
    //     ss << "[";
    //     for (shared_ptr<const StringAction> act : *vec) {
    //         ss << act->action << ",";
    //     }
    //     ss << "]";
    //     return ss.str();
    // }

    // string get_vector_string(shared_ptr<ActionVector> vec) {
    //     shared_ptr<StringActionVector> cast_vec = make_shared<StringActionVector>();
    //     for (shared_ptr<const Action> act : *vec) {
    //         cast_vec->push_back(static_pointer_cast<const StringAction>(act));
    //     }
    //     return get_vector_string(cast_vec);
    // }

    // /**
    //  * Helper to print out distributions.
    //  */
    // string get_map_string(shared_ptr<IntPairStateDistr> mp) {
    //     stringstream ss;
    //     ss << "{";
    //     for (pair<shared_ptr<const IntPairState>,double> pr : *mp) {
    //         ss << get_state_string(pr.first) << ":" << pr.second << ",";
    //     }
    //     ss << "}";
    //     return ss.str();
    // }

    // string get_map_string(shared_ptr<ObservationDistr> mp) {
    //     shared_ptr<IntPairStateDistr> cast_mp = make_shared<IntPairStateDistr>();
    //     for (pair<shared_ptr<const Observation>,double> pr : *mp) {
    //         cast_mp->insert_or_assign(static_pointer_cast<const IntPairState>(pr.first), pr.second);
    //     }
    //     return get_map_string(cast_mp);
    // }

    /**
     * TODO: document
     */
    void run_thts_env_tests() {
        cout << "----------" << endl;
        cout << "Testing ThtsEnv using a test subclass." << endl;
        cout << "----------" << endl;
        
        // Test all the functions of the test thts env by iterating through a few states with fixed actions
        TestThtsEnv env(1);

        shared_ptr<const IntPairState> init_state = env.get_initial_state();

        cout << *(init_state) << endl;
        cout << *(env.get_valid_actions(init_state)) << endl;
        cout << env.is_sink_state(init_state) << endl << endl;

        shared_ptr<const StringAction> act = make_shared<const StringAction>(StringAction("right"));
        shared_ptr<const IntPairState> next_state = env.sample_transition_distribution(init_state, act);

        cout << env.get_reward(init_state, act) << endl;
        cout << *(next_state) << endl;
        cout << *(env.get_valid_actions(next_state)) << endl;
        cout << env.is_sink_state(next_state) << endl << endl;

        act = make_shared<const StringAction>(StringAction("down"));
        next_state = env.sample_transition_distribution(next_state, act);

        cout << env.get_reward(init_state, act) << endl;
        cout << *(next_state) << endl;
        cout << *(env.get_valid_actions(next_state)) << endl;
        cout << env.is_sink_state(next_state) << endl << endl;

        // Also check that getting the entirer transition distribution happens correctly
        shared_ptr<IntPairStateDistr> distribution = env.get_transition_distribution(init_state, act);
        cout << *(distribution) << endl << endl;

        // Repeat all of the above, but using the interface, AND, only printing out at the end to check shared memory 
        // is allocated correctly. (Should get some garbage if not).
        shared_ptr<const State> state_0 = env.get_initial_state_itfc();
        shared_ptr<ActionVector> actions_0 = env.get_valid_actions_itfc(state_0);
        shared_ptr<const StringAction> act_string_0 = make_shared<const StringAction>(StringAction("right"));

        shared_ptr<const Action> act_0 = static_pointer_cast<const Action>(act_string_0);
        shared_ptr<const Observation> obsv_1 = env.sample_transition_distribution_itfc(state_0, act_0);
        shared_ptr<const State> state_1 = static_pointer_cast<const State>(obsv_1);
        shared_ptr<ActionVector> actions_1 = env.get_valid_actions_itfc(state_1);
        shared_ptr<const StringAction> act_string_1 = make_shared<const StringAction>(StringAction("down"));
        shared_ptr<const Action> act_1 = static_pointer_cast<const Action>(act_string_1);

        shared_ptr<const Observation> obsv_2 = env.sample_transition_distribution_itfc(state_1, act_1);
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

    }
}