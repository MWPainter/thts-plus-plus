#include "benchmark.h"

#include "helper_templates.h"
#include "go/go_env.h"
#include "go/go_state_action.h"

#include "KataGo/cpp/game/board.h"
#include "KataGo/cpp/neuralnet/nninputs.h"

#include <cstdlib>
#include <iostream>
#include <memory>

using namespace std;
using namespace thts;


// Overload printing goaction here, because generic printing via thts::Action didn't work because pointers
namespace std {
    ostream& operator<<(ostream& os, const shared_ptr<const GoAction>& action) {
        os << action->loc;
        return os;
    }
}


namespace thts_test {
    int NUM_ACTIONS = 100;

    void go_state_benchmark() {
        // Katago needs a global init for hashing
        // Also init score tables for using kata go score value computation
        Board::initHash(); 
        ScoreValue::initTables();

        // Just make random moves and see how many we can get through
        GoEnv go_env(19,7.5);
        shared_ptr<const GoState> cur_state = go_env.get_initial_state();
        int i = 0;
        while (i < NUM_ACTIONS) {
            if (go_env.is_sink_state(cur_state)) {
                cur_state = go_env.get_initial_state();
            }
            shared_ptr<GoActionPolicy> policy = go_env.get_policy_from_nn(cur_state);
            double heuristic_val = go_env.get_heuristic_val_from_nn(cur_state);
            shared_ptr<GoActionVector> action_list = go_env.get_valid_actions(cur_state);
            int idx = rand() % action_list->size();
            cur_state = go_env.sample_transition_distribution(cur_state, action_list->at(idx));
            i++;
            if (i % (NUM_ACTIONS/10) == 0) {
                cout << "Performed " << i << " go moves." << endl;
                cout << cur_state->get_pretty_print_string() << endl;
                cout << "Heuristic val: " << heuristic_val << endl;
                cout << "Policy: " << helper::unordered_map_pretty_print_string(*policy) << endl;
            }
        }
    }
}