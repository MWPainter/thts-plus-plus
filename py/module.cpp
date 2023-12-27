#include "test_env.h"

#include "py/py_thts_env.h"

#include "algorithms/est/est_decision_node.h"
#include "thts_decision_node.h"
#include "thts.h"

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

#include <chrono>
#include <iostream>

using namespace std;
using namespace thts;
using namespace thts::python;
namespace py = pybind11;
using namespace py::literals;


int add(int i, int j) {
    return i + j;
}

int bts_test(double alpha) {
    // // startup python interpreter
    // // Only need interpreter object when running a c++ program and need python things
    // // Here we're a function for the python module => interpreter already exists
    // py::scoped_interpreter guard{};

    // params
    int env_size = 3;
    double stay_prob = 0.1;
    int num_trials = 10000;
    int print_tree_depth = 2;
    int num_threads = 4;

    // Make py env (making a py::object of python thts env, and pass into constructor)
    // py::module_ py_thts_env_module = py::module_::import("test_env");
    // py::object py_thts_env = py_thts_env_module.attr("PyTestThtsEnv")(3, 0.1);
    // shared_ptr<thts::python::PyThtsEnv> thts_env = make_shared<thts::py::PyThtsEnv>(py_thts_env);
    shared_ptr<thts::python::TestThtsEnv> thts_env = make_shared<thts::python::TestThtsEnv>(env_size, stay_prob);

    // Make thts manager with the py env (same as c++ (use unit tests))
    DentsManagerArgs manager_args(thts_env);
    manager_args.seed = 60415;
    manager_args.max_depth = env_size * 4;
    manager_args.mcts_mode = false;
    manager_args.temp = alpha;
    shared_ptr<DentsManager> manager = make_shared<DentsManager>(manager_args);
    shared_ptr<EstDNode> root_node = make_shared<EstDNode>(manager, thts_env->get_initial_state_itfc(), 0, 0);
    ThtsPool bts_pool(manager, root_node, num_threads);

    // Run thts trials (same as c++)
    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();
    bts_pool.run_trials(num_trials);
    std::chrono::duration<double> dur = chrono::system_clock::now() - start_time;

    // Print out a tree (same as c++)
    cout << "EST with " << num_threads << " threads (took " << dur.count() << ")";
    if (print_tree_depth > 0){
        cout << " and looks like:\n";
        cout << root_node->get_pretty_print_string(print_tree_depth) << endl;
    } else {
        cout << endl;
    }

    // Return success
    return 0;
}

PYBIND11_MODULE(thts, m) {
    // Module docstring
    m.doc() = "python module to access the THTS++ library";

    // This is an example module call for BTS
    // TODO: add py::object argument to take pass in a custom env
    // TODO: add custom pybind objects to return (top levels of the) search tree
    m.def("add", &add, "A function that adds two numbers");
    // m.def("bts_test",
    //         &bts_test,
    //         "Function taking only alpha value for bts and running it in small test env",
    //         py::arg("alpha"));
};