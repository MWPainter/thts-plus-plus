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

    // // test 7 - check my lock guard works as intended
    // {
    //     thts::python::helper::GilReenterantLockGuard lg;
    //     py::print("h");

    //     // text 8 - check it's re-enterant
    //     {
    //         thts::python::helper::GilReenterantLockGuard lg;
    //         py::print("i");
    //     }
    // }
    // // py::print("j");
    // // // This print j causes a segfault, because don't have lockguard anymore
}

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
        thts_env = make_shared<thts::python::PyThtsEnv>(make_shared<py::object>(py_thts_env), false);
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
    auto gstate = thts::python::helper::lock_gil();
    shared_ptr<EstDNode> root_node = make_shared<EstDNode>(manager, thts_env->get_initial_state_itfc(), 0, 0);
    thts::python::helper::unlock_gil(gstate);
    shared_ptr<ThtsPool> bts_pool;
    if (use_python_env) {
        bts_pool = make_shared<PyThtsPool>(manager, root_node, num_threads);
    } else {
        bts_pool = make_shared<ThtsPool>(manager, root_node, num_threads);
    }

    // Run thts trials (same as c++)
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

void bts_test_cpp_env(double alpha) {
    // Assume we already have an interpreter, and need to release gil
    // release gil and use lockguard to protect ourselves
    py::gil_scoped_release rel;
    bts_test(alpha, false);
}

void bts_test_py_env(double alpha) {
    // Assume we already have an interpreter, and need to release gil
    // release gil and use lockguard to protect ourselves
    py::gil_scoped_release rel;
    bts_test(alpha, true);
}

// // Setup python interpreter in CPython
// // UNUSED, doesn't work in current state either, but didnt want to get rid of it
// void setup_python_interpreter() {
//     PyStatus status;

//     // config
//     // get default python config
//     // run in an isolated environment (seems like a good idea)
//     // add <working_directory>/thts-plus-plus/py into search paths
//     PyConfig config;
//     PyConfig_InitPythonConfig(&config);

//     status = PyConfig_Read(&config);
//     if (PyStatus_Exception(status)) {
//         throw runtime_error(status.err_msg);
//     }

//     config.isolated = 1;    

//     config.module_search_paths_set = 1;
//     std::string py_dir_path = std::filesystem::current_path().string() + "/py";
//     std::wstring wstring_py_dir_path = std::wstring(py_dir_path.begin(), py_dir_path.end());
//     status = PyWideStringList_Append(&config.module_search_paths, wstring_py_dir_path.c_str());

//     // error handling in C is annoying
//     if (PyStatus_Exception(status)) {
//         throw runtime_error(status.err_msg);
//     }
    
//     // make interpreter with config
//     status = Py_InitializeFromConfig(&config);

//     // error handling in C is annoying
//     if (PyStatus_Exception(status)) {
//         throw runtime_error(status.err_msg);
//     }
//     PyConfig_Clear(&config);
// }

struct MorePybindPlay {
    shared_ptr<thts::python::PyThtsEnv> thts_env;
    std::thread workers[2];
    std::mutex lock;
    bool stop;

    MorePybindPlay() {
        py::gil_scoped_acquire acq;
        py::module_ py_thts_env_module = py::module_::import("test_env");
        py::object py_thts_env = py_thts_env_module.attr("PyTestThtsEnv")(2, 0.0);
        thts_env = make_shared<thts::python::PyThtsEnv>(make_shared<py::object>(py_thts_env), false);

        workers[0] = thread(&MorePybindPlay::run_thread, this, 0);
        workers[1] = thread(&MorePybindPlay::run_thread, this, 1);

        stop = false;
    }

    virtual ~MorePybindPlay() {
        py::gil_scoped_acquire acq;
        thts_env.reset();
    }

    void run_thread(int tid) {
        thts::python::helper::lock_gil();
        PyInterpreterConfig config = {
            .use_main_obmalloc = 0,
            .allow_fork = 0,
            .allow_exec = 0,
            .allow_threads = 1,
            .allow_daemon_threads = 0,
            .check_multi_interp_extensions = 1,
            .gil = PyInterpreterConfig_OWN_GIL,
        };
        PyThreadState *tstate;
        Py_NewInterpreterFromConfig(&tstate, &config);
        if (tstate == NULL) {
            throw runtime_error("Error starting subinterpreter");
        }

        do_work(tid);
    };

    void do_work(int tid) {
        RandManager rand;
        shared_ptr<const PyState> state = thts_env->get_initial_state();
        while (!stop) {
            std::shared_ptr<PyActionVector> actions = thts_env->get_valid_actions(state);
            std::shared_ptr<const PyAction> action = actions->at(0);
            state = thts_env->sample_transition_distribution(state, action, rand);
            cout << thts_env->get_reward(state, action) << " frm " << tid << endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    void stahp() {
        stop = true;
        workers[0].join();
        workers[1].join();
    }
};

// C++ entry point for debugging
int main(int argc, char *argv[]) {
    // Make python interpreter
    py::scoped_interpreter guard;
    py::gil_scoped_release rel;
    // TODO: setup env variables so dont need the setup_python_dev.sh

    // // testing pybind out
    // pybind11_testing(); // change this to call want to debug, or make own main if making C++ ex and not making py lib

    // // testing python subinterpreters
    // MorePybindPlay play;
    // std::this_thread::sleep_for(std::chrono::seconds(1));
    // play.stahp();
    
    // testing bts with python env with c++ entrypoint for debugging
    bool bts_alpha = 1.0;
    bool use_python_env = true;
    bts_test(bts_alpha, use_python_env); 

    return 0;
}

PYBIND11_MODULE(thts, m) { 

    // Module docstring
    m.doc() = "python module to access the THTS++ library";

    // TODO: find some way to add env variables at the beginning of any thts module calls

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