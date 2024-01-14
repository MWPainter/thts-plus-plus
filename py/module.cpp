#include "module.h"

#include "test_env.h"

#include "py/pickle_wrapper.h"
#include "py/py_multiprocessing_thts_env.h"
#include "py/py_thts_env.h"
#include "py/py_helper.h"
#include "py/py_thts.h"
#include "py/shared_mem_wrapper.h"

#include "algorithms/est/est_decision_node.h"
#include "thts_decision_node.h"

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <thread>

#include <sys/sem.h>
#include <unistd.h>

using namespace std;
using namespace thts;
using namespace thts::python;
namespace py = pybind11;
using namespace py::literals;


// test pickling + converting pickled objects to and from C++ strings
void pickle_test() {
    py::module_ py_pickle_module = py::module_::import("pickle");
    py::dict py_dict;
    py_dict["tid"] = 1;
    py_dict["somethingelse"] = "tid";

    py::object py_serialise_fn = py_pickle_module.attr("dumps");
    py::object py_dict_serialised = py_serialise_fn(py_dict);
    py::print("Python print:");
    py::print(py_dict_serialised);
    py::print(py_dict);
    py::print();

    string serialised_dict_stirng = py_dict_serialised.cast<string>();
    cout << "C++ print:" << endl << serialised_dict_stirng << endl << endl;

    py::object py_deserialise_fn = py_pickle_module.attr("loads");
    py::object py_dict_deserialised = py_deserialise_fn(py_dict_serialised);
    py::object py_dict_serialised_from_cpp = py::bytes(serialised_dict_stirng);
    py::object py_dict_deserialised_from_cpp = py_deserialise_fn(py_dict_serialised_from_cpp);
    py::print("Python deserialised print:");
    py::print(py_dict_deserialised);
    py::print(py_dict_deserialised_from_cpp);
}

// testing unix semiphores interface we defined
void sem_test() {
    key_t key = thts::python::helper::get_unix_key(0);
    int semid = thts::python::helper::init_sem(key, 1);

    pid_t pid = fork();

    // child program
    if (pid == 0 ) {
        for (int i=0; i<10; i++) {
            thts::python::helper::acquire_sem(semid, 0);
            cout << "In child, loop" << i << endl;
            thts::python::helper::release_sem(semid, 0);
        }
        exit(0);
    }

    // Parent program
    for (int i=0; i<10; i++) {
        thts::python::helper::acquire_sem(semid, 0);
        cout << "In parent, loop" << i << endl;
        thts::python::helper::release_sem(semid, 0);
    }

    thts::python::helper::acquire_sem(semid, 0);
    thts::python::helper::destroy_sem(semid);
}

// testing that python interpreter is copied on a fork
void pickle_multiproc_test() {
    key_t key = thts::python::helper::get_unix_key(0);
    int semid = thts::python::helper::init_sem(key, 1);

    // acquire before, so sem is = 0
    thts::python::helper::acquire_sem(semid, 0);

    // spawn child
    pid_t pid = fork();

    // child program
    if (pid == 0 ) {
        pickle_test();
        thts::python::helper::release_sem(semid, 0); // release to signal to parent
        exit(0);
    }

    // Parent program, wait for child to release us and cleanup
    thts::python::helper::acquire_sem(semid, 0);
    thts::python::helper::destroy_sem(semid);
}

void pickle_subinterpret_test_loop(int tid, py::module_& py_pickle_module, mutex& lock) {
    // subinterpreter
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

    for (int i=0; i<1000; i++) {
        lock.lock();
        py::dict py_dict;
        py_dict["tid"] = 1;
        py_dict["somethingelse"] = "tid";
        py::object py_serialise_fn = py_pickle_module.attr("dumps");
        py::object py_dict_serialised = py_serialise_fn(py_dict);  
        string serialised_dict_stirng = py_dict_serialised.cast<string>();   
        py::object py_deserialise_fn = py_pickle_module.attr("loads");
        py::object py_dict_deserialised = py_deserialise_fn(py_dict_serialised);
        py::object py_dict_serialised_from_cpp = py::bytes(serialised_dict_stirng);
        py::object py_dict_deserialised_from_cpp = py_deserialise_fn(py_dict_serialised_from_cpp);   
        if ((i%100) == 0) { 
            // lock.lock();
            cout << "finished loop " << i << " in thread " << tid << endl;
            // lock.unlock();
        }
        lock.unlock();
    }

}

/**
 * Pickle works with subinterpreters
 * But
 * Pickle needs to be protected from concurrency still
*/
void pickle_subinterpret_test() {
    py::module_ py_pickle_module = py::module_::import("pickle");
    py::gil_scoped_release rel;
    mutex lock;
    thread t0(&pickle_subinterpret_test_loop, 0, std::ref(py_pickle_module), std::ref(lock));
    thread t1(&pickle_subinterpret_test_loop, 1, std::ref(py_pickle_module), std::ref(lock));
    t0.join();
    t1.join();
}

// testing setting up shared memory 
// updating an integer in a piece of shared memory
// this requires semaphores, so this is an adaption of the semaphore test
void shared_mem_test() {
    key_t key = thts::python::helper::get_unix_key(0);
    int semid = thts::python::helper::init_sem(key, 1);
    int shmid = thts::python::helper::init_shared_mem(key,1024);
    int* shared_int = (int*) thts::python::helper::get_shared_mem_ptr(shmid);
    *shared_int = 0;

    pid_t pid = fork();

    // child program
    if (pid == 0 ) {
        for (int i=0; i<10; i++) {
            thts::python::helper::acquire_sem(semid, 0);
            cout << "In child, loop " << i << ", shared_int " << *shared_int << endl;
            (*shared_int)++;
            thts::python::helper::release_sem(semid, 0);
        }
        exit(0);
    }

    // Parent program
    for (int i=0; i<10; i++) {
        thts::python::helper::acquire_sem(semid, 0);
        cout << "In parent, loop " << i << ", shared_int " << *shared_int << endl;
        (*shared_int)++;
        thts::python::helper::release_sem(semid, 0);
    }

    thts::python::helper::acquire_sem(semid, 0);
    thts::python::helper::destroy_sem(semid);
    thts::python::helper::destroy_shared_mem(shmid);
}

void shared_mem_wrapper_test() {
    SharedMemWrapper smw(0,8*1024);
    pid_t pid = fork();
    if (pid == 0) {
        smw.server_wait_for_rpc_call();
        int rpc_id = smw.rpc_id;
        int num_args = smw.num_args;
        string arg1 = smw.args[0];
        string arg2 = smw.args[1];
        cout << "Recieved rpc call:" << endl << rpc_id << endl << num_args << endl << arg1 << endl << arg2 << endl;
        smw.rpc_id = 0;
        smw.num_args = 1;
        smw.args[0] = "RPCRESULT";
        smw.server_send_rpc_call_result();
        exit(0);
    }

    smw.rpc_id = 3;
    smw.num_args = 2;
    smw.args[0] = "RPCARG1";
    smw.args[1] = "RPCARG2";
    smw.make_rpc_call();

    cout << "Recieved rpc result:" << endl << smw.rpc_id << endl << smw.num_args << endl << smw.args[0] << endl;
}




void bts_test(double alpha, bool use_python_env) {
    // release gil
    py::gil_scoped_release rel;

    // params
    int env_size = 3;
    double stay_prob = 0.0;
    int num_trials = 10000;
    int print_tree_depth = 2;
    int num_threads = 4;

    // Make py env (making a py::object of python thts env, and pass into constructor)
    shared_ptr<ThtsEnv> thts_env;
    if (use_python_env) {
        py::gil_scoped_acquire acq;
        py::module_ py_thts_env_module = py::module_::import("test_env"); 
        py::object py_thts_env = py_thts_env_module.attr("PyTestThtsEnv")(env_size, stay_prob);
        shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
        thts_env = make_shared<PyMultiprocessingThtsEnv>(pickle_wrapper, make_shared<py::object>(py_thts_env));
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
        args.num_envs = num_threads;
        // args.num_envs = 1;
        manager = make_shared<DentsManager>(args);

        // Setup python servers
        if (use_python_env) {
            for (int i=0; i<args.num_envs; i++) {
                PyMultiprocessingThtsEnv& py_mp_env = (PyMultiprocessingThtsEnv&) *manager->thts_env(i);
                py_mp_env.start_python_server(i);
            }
        }
        
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
    manager.reset();
    manager_args.reset();
    thts_env.reset();
    bts_pool.reset();
    root_node.reset();
}

// C++ entry point for debugging
int main(int argc, char *argv[]) {
    py::scoped_interpreter guard;

    // pickle_test();
    // sem_test();
    // pickle_multiproc_test();
    // pickle_subinterpret_test();
    // shared_mem_test();
    // shared_mem_wrapper_test();

    bool bts_alpha = 1.0;
    bool use_python_env = true;
    bts_test(bts_alpha, use_python_env); 

    return 0;
}

PYBIND11_MODULE(thts, m) { 

    // Module docstring
    m.doc() = "python module to access the THTS++ library";
};