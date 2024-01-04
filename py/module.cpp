#include "module.h"

#include "test_env.h"

#include "py/gil_helpers.h"
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

void pybind11_testing() {
    py::print("a");

    // // test 1 = dont have gill = error when use pybind?
    // py::gil_scoped_release rel1;
    // py::print("b");
    // // result of running this is a segfault

    // test 2 = testing if gil is reenterant
    py::gil_scoped_acquire acc1;
    py::gil_scoped_acquire acc2;
    py::print("b");
    // this prints b

    // test 3 = testing if ref counting on releases
    py::gil_scoped_release rel2;
    // py::print("c");
    // // trying to print c here fails -> gil released

    // // test 4 - can release again without reprocussion
    // py::gil_scoped_release rel3;
    // // nope, this caused a python error

    // test 5 - still have lock after acquire/require goes out of scope?
    {
        py::gil_scoped_acquire acc3;
        py::print("d");
    }
    // py::print("e");
    // // This print e causes segfault, so don't have gil anymore after acquire goes out of scope

    // test 6 - gil_scoped_acquire going out of scope doesn't cause the gil to be released
    py::gil_scoped_acquire acc4;
    {
        py::gil_scoped_acquire acc5;
        py::print("f");
    }
    py::print("g");
    py::gil_scoped_release rel4;

    // test 7 - check my lock guard works as intended
    {
        thts::python::helpers::GilReenterantLockGuard lg;
        py::print("h");

        // text 8 - check it's re-enterant
        {
            thts::python::helpers::GilReenterantLockGuard lg;
            py::print("i");
        }
    }
    // py::print("j");
    // // This print j causes a segfault, because don't have lockguard anymore
}

void bts_test(double alpha, bool use_python_env) {
    // Assume we already have an interpreter, and need to release gil
    // release gil and use lockguard to protect ourselves
    py::gil_scoped_release rel;

    // params
    int env_size = 3;
    double stay_prob = 0.1;
    int num_trials = 1000;
    int print_tree_depth = 2;
    int num_threads = 1;

    // Make py env (making a py::object of python thts env, and pass into constructor)
    shared_ptr<ThtsEnv> thts_env;
    if (use_python_env) {
        thts::python::helpers::GilReenterantLockGuard lg;
        py::module_ py_thts_env_module = py::module_::import("test_env");
        py::object py_thts_env = py_thts_env_module.attr("PyTestThtsEnv")(env_size, stay_prob);
        thts_env = make_shared<thts::python::PyThtsEnv>(py_thts_env);
    } else {
        thts_env = make_shared<thts::python::TestThtsEnv>(env_size, stay_prob);
    }

    // Make thts manager with the py env (same as c++ (use unit tests))
    shared_ptr<DentsManagerArgs> manager_args = make_shared<DentsManagerArgs>(thts_env);
    manager_args->seed = 60415;
    manager_args->max_depth = env_size * 4;
    manager_args->mcts_mode = false;
    manager_args->temp = alpha;
    shared_ptr<DentsManager> manager = make_shared<DentsManager>(*manager_args);
    shared_ptr<EstDNode> root_node = make_shared<EstDNode>(manager, thts_env->get_initial_state_itfc(), 0, 0);
    shared_ptr<ThtsPool> bts_pool = make_shared<ThtsPool>(manager, root_node, num_threads);

    // Run thts trials (same as c++)
    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();
    bts_pool->run_trials(num_trials);
    std::chrono::duration<double> dur = chrono::system_clock::now() - start_time;

    // Print out a tree (same as c++)
    cout << "EST with " << num_threads << " threads (took " << dur.count() << ")";
    if (print_tree_depth > 0){
        cout << " and looks like:\n";
        cout << root_node->get_pretty_print_string(print_tree_depth) << endl;
    } else {
        cout << endl;
    }

    // Force tree destructors to be called (to clean up python objects with the gil)
    // resetting smart pointers should refcount to zero and call destructor
    // A bit annoying having to make everything a smart pointer to call destructors with gil, but mech
    thts::python::helpers::GilReenterantLockGuard lg;
    bts_pool.reset();
    root_node.reset();
    manager.reset();
    manager_args.reset();
    thts_env.reset();
}

void bts_test_cpp_env(double alpha) {
    bts_test(alpha, false);
}

void bts_test_py_env(double alpha) {
    bts_test(alpha, true);
}

// C++ entry point for debugging
int main(int argc, char *argv[]) {
    py::scoped_interpreter guard{}; // make a python interpreter
    // pybind11_testing(); // change this to call want to debug, or make own main if making C++ ex and not making py lib
    bts_test(1.0, false); 
    return 0;
}

PYBIND11_MODULE(thts, m) { 

    // Module docstring
    m.doc() = "python module to access the THTS++ library";

    // This is an example module call for BTS
    // TODO: add py::object argument to take pass in a custom env
    // TODO: add custom pybind objects to return (top levels of the) search tree
    m.def("add", &add, "A function that adds two numbers");
    m.def("pybind11_testing", &pybind11_testing, "Playing with pybind");
    m.def("bts_test_cpp_env",
            &bts_test_cpp_env,
            "Function taking only alpha value for bts and running it in small (C++) test env",
            py::arg("alpha"));
    m.def("bts_test_py_env",
            &bts_test_py_env,
            "Function taking only alpha value for bts and running it in small (python) test env",
            py::arg("alpha"));
};






// // C++ entry point for debugging
// int main(int argc, char *argv[]) {
//     py::scoped_interpreter guard{}; // make a python interpreter
//     cout << "1" << endl;
//     cout << "2" << endl;
//     return 0;
// }

// int add(int i, int j) {
//     return i + j;
// }

// PYBIND11_MODULE(thts, m) { // build for c++ main
//     m.doc() = "pybind11 example plugin"; // optional module docstring

//     m.def("add", &add, "A function that adds two numbers");
// }