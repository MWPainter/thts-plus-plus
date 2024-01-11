#include "module.h"

#include "test_env.h"

#include "py/py_thts_env.h"
#include "py/py_helper.h"
#include "py/py_thts.h"

#include "algorithms/est/est_decision_node.h"
#include "thts_decision_node.h"

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <thread>

using namespace std;
using namespace thts;
using namespace thts::python;
namespace py = pybind11;
using namespace py::literals;



void bts_test(double alpha, bool use_python_env) {

    // params
    int env_size = 3;
    double stay_prob = 0.0;
    int num_trials = 100000;
    int print_tree_depth = 2;
    int num_threads = 4;

    // Make py env (making a py::object of python thts env, and pass into constructor)
    shared_ptr<ThtsEnv> thts_env;
    if (use_python_env) {
        py::gil_scoped_acquire acq;
        py::module_ py_thts_env_module = py::module_::import("test_env"); 
        py::object py_thts_env = py_thts_env_module.attr("PyTestThtsEnv")(env_size, stay_prob);
        thts_env = make_shared<PyThtsEnv>(make_shared<py::object>(py_thts_env), true);
    } else {
        thts_env = make_shared<thts::python::TestThtsEnv>(env_size, stay_prob);
    }

    // Make thts manager with the py env (same as c++ (use unit tests))
    // But protect with GIL for any python ops in creating things
    shared_ptr<DentsManagerArgs> manager_args;
    shared_ptr<DentsManager> manager;
    shared_ptr<EstDNode> root_node;
    shared_ptr<ThtsPool> bts_pool;
    {
        py::gil_scoped_acquire acq;

        DentsManagerArgs args(thts_env);
        args.seed = 60415;
        args.max_depth = env_size * 4;
        args.mcts_mode = false;
        args.temp = alpha;
        args.num_threads = num_threads;
        args.num_envs = 1;
        manager = make_shared<DentsManager>(args);
        
        root_node = make_shared<EstDNode>(
            manager, thts_env->get_initial_state_itfc(), 0, 0);
        if (use_python_env) {
            bts_pool = make_shared<PyThtsPool>(manager, root_node, num_threads);
        } else {
            bts_pool = make_shared<ThtsPool>(manager, root_node, num_threads);
        }
    }

    // Run thts trials (same as c++)
    // Needs to not have the gil, so threads can grab it any make interpreters
    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();
    bts_pool->run_trials(num_trials);
    std::chrono::duration<double> dur = chrono::system_clock::now() - start_time;

    // Print out a tree (same as c++)
    // Make sure have gil, because getting pretty print string using python objects
    py::gil_scoped_acquire acq;
    cout << "EST with " << num_threads << " threads (took " << dur.count() << ")";
    if (print_tree_depth > 0) {
        cout << " and looks like:\n";
        cout << root_node->get_pretty_print_string(print_tree_depth) << endl;
    } else {
        cout << endl;
    }

    // Force tree destructors to be called (to clean up python objects with the gil)
    // resetting smart pointers should refcount to zero and call destructor
    // A bit annoying having to make everything a smart pointer to call destructors with gil, but mech
    bts_pool.reset();
    root_node.reset();
    manager.reset();
    manager_args.reset();
    thts_env.reset();
}

// C++ entry point for debugging
int main(int argc, char *argv[]) {
    py::scoped_interpreter guard;
    py::gil_scoped_release rel;

    bool bts_alpha = 1.0;
    bool use_python_env = true;
    bts_test(bts_alpha, use_python_env); 

    return 0;
}

PYBIND11_MODULE(thts, m) { 

    // Module docstring
    m.doc() = "python module to access the THTS++ library";
};