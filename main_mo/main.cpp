#include "main_mo/run_id.h"
#include "main_mo/run_expr.h"
#include "main_mo/val.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "main_mo/envs/tree_env.h"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

#include <iostream>
#include "py/py_helper.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        throw runtime_error("Expecting exactly two arguments: [eval|opt] [expr_id], specifying if we want to run an "
                            "eval experiment, or perform hyperparamter optimisation.");
    }

    if (string(argv[1]) == "eval") {
        shared_ptr<vector<RunID>> run_ids = thts::get_run_ids_from_expr_id_prefix(argv[2]);
        thts::run_exprs(run_ids);
    } else if (string(argv[1]) == "opt") {  
        thts::run_hp_opt(argv[2]);
    } else if (string(argv[1]) == "val") {
        thts::run_valgrind_debugging(stoi(string(argv[2])));
    }

    return 0;
}

// /**
//  * Testing tree env
//  */

// int main(int argc, char* argv[]) {

//     pybind11::scoped_interpreter py_interpreter;
//     pybind11::gil_scoped_release release;

//     ToyTreeEnv env1(2,10,5,false);
//     int i = 0;
//     for (Eigen::ArrayXd& rew : env1.reward_vectors) {
//         cout << "rew " << i++ << ": " << rew << endl << endl;
//     }

//     ToyTreeEnv env2(2,5,5,false);
//     i = 0;
//     for (Eigen::ArrayXd& rew : env2.reward_vectors) {
//         cout << "rew " << i++ << ": " << rew << endl << endl;
//     }

//     ToyTreeEnv env3(2,2,5,false);
//     i = 0;
//     for (Eigen::ArrayXd& rew : env3.reward_vectors) {
//         cout << "rew " << i++ << ": " << rew << endl << endl;
//     }

//     RandManager manager;
//     ThtsContext ctx;
//     shared_ptr<const State> cur_state = env1.get_initial_state_itfc();
//     shared_ptr<const Action> act = static_pointer_cast<const Action>(make_shared<const IntAction>(9));
//     for (i=0; i<5; i++) {
//         cout << i << "th dense reward: " << env1.get_mo_reward_itfc(cur_state,act,ctx) << endl << endl;
//         cur_state = env1.sample_transition_distribution_itfc(cur_state,act,manager,ctx);
//     }
//     cout << "Final state is sink? " << env1.is_sink_state_itfc(cur_state,ctx) << endl << endl;

//     ToyTreeEnv env4(2,10,5,true);
//     cur_state = env4.get_initial_state_itfc();
//     for (i=0; i<5; i++) {
//         cout << i << "th sparse reward: " << env4.get_mo_reward_itfc(cur_state,act,ctx) << endl << endl;
//         cur_state = env4.sample_transition_distribution_itfc(cur_state,act,manager,ctx);
//     }
//     cout << "Final state is sink? " << env4.is_sink_state_itfc(cur_state,ctx) << endl << endl;
// }