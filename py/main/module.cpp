#include "module.h"

#include "test_env.h"

#include "py/gym_multiprocessing_thts_env.h"
#include "py/mo_gym_multiprocessing_thts_env.h"
#include "py/mo_py_thts.h"
#include "py/pickle_wrapper.h"
#include "py/py_multiprocessing_thts_env.h"
#include "py/py_thts_env.h"
#include "py/py_helper.h"
#include "py/py_thts.h"
#include "py/shared_mem_wrapper.h"

#include "algorithms/est/est_decision_node.h"
#include "thts_decision_node.h"

#include "test/mo/test_mo_thts_env.h"

#include "mo/chmcts_manager.h"
#include "mo/chmcts_decision_node.h"
#include "mo/czt_manager.h"
#include "mo/czt_decision_node.h"
#include "mo/smt_bts_manager.h"
#include "mo/smt_bts_decision_node.h"
#include "mo/smt_dents_manager.h"
#include "mo/smt_dents_decision_node.h"
#include "mo/mo_mc_eval.h"
#include "mo/mo_thts.h"
#include "mo/mo_thts_context.h"
#include "mo/mo_helper.h"

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <mutex>
#include <thread>

#include <sys/sem.h>
#include <unistd.h>
#include <stdlib.h>

// #include "lemon/lp.h"

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
    key_t key = thts::python::helper::get_unix_key();
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
    key_t key = thts::python::helper::get_unix_key();
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

/**
 * Getting rid of subinterpreter stuff
 * lock_gil used to call the cpython lock gil function, that was it
 * Keeping this though just incase ever want to use subinterpreters again?
 * Probably not, but better to avoid the all of the pain of working this out from scratch again in the odd case do want
 */
// void pickle_subinterpret_test_loop(int tid, py::module_& py_pickle_module, mutex& lock) {
//     // subinterpreter
//     thts::python::helper::lock_gil();
//     PyInterpreterConfig config = {
//         .use_main_obmalloc = 0,
//         .allow_fork = 0,
//         .allow_exec = 0,
//         .allow_threads = 1,
//         .allow_daemon_threads = 0,
//         .check_multi_interp_extensions = 1,
//         .gil = PyInterpreterConfig_OWN_GIL,
//     };
//     PyThreadState *tstate;
//     Py_NewInterpreterFromConfig(&tstate, &config);
//     if (tstate == NULL) {
//         throw runtime_error("Error starting subinterpreter");
//     }

//     for (int i=0; i<1000; i++) {
//         lock.lock();
//         py::dict py_dict;
//         py_dict["tid"] = 1;
//         py_dict["somethingelse"] = "tid";
//         py::object py_serialise_fn = py_pickle_module.attr("dumps");
//         py::object py_dict_serialised = py_serialise_fn(py_dict);  
//         string serialised_dict_stirng = py_dict_serialised.cast<string>();   
//         py::object py_deserialise_fn = py_pickle_module.attr("loads");
//         py::object py_dict_deserialised = py_deserialise_fn(py_dict_serialised);
//         py::object py_dict_serialised_from_cpp = py::bytes(serialised_dict_stirng);
//         py::object py_dict_deserialised_from_cpp = py_deserialise_fn(py_dict_serialised_from_cpp);   
//         if ((i%100) == 0) { 
//             // lock.lock();
//             cout << "finished loop " << i << " in thread " << tid << endl;
//             // lock.unlock();
//         }
//         lock.unlock();
//     }
// }

/**
 * Pickle works with subinterpreters
 * But
 * Pickle needs to be protected from concurrency still
*/
// void pickle_subinterpret_test() {
//     py::module_ py_pickle_module = py::module_::import("pickle");
//     py::gil_scoped_release rel;
//     mutex lock;

//     thread t0(&pickle_subinterpret_test_loop, 0, std::ref(py_pickle_module), std::ref(lock));
//     thread t1(&pickle_subinterpret_test_loop, 1, std::ref(py_pickle_module), std::ref(lock));
//     t0.join();
//     t1.join();
// }

// testing setting up shared memory 
// updating an integer in a piece of shared memory
// this requires semaphores, so this is an adaption of the semaphore test
void shared_mem_test() {
    key_t key = thts::python::helper::get_unix_key();
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
    throw runtime_error("shared mem wrapper tests need upstring for changes made in refactor");
    // SharedMemWrapper smw(0,8*1024);
    // pid_t pid = fork();
    // if (pid == 0) {
    //     smw.server_wait_for_rpc_call();
    //     int rpc_id = smw.rpc_id;
    //     int num_args = smw.num_args;
    //     string arg1 = smw.args[0];
    //     string arg2 = smw.args[1];
    //     cout << "Recieved rpc call:" << endl << rpc_id << endl << num_args << endl << arg1 << endl << arg2 << endl;
    //     smw.rpc_id = 0;
    //     smw.num_args = 1;
    //     smw.args[0] = "RPCRESULT";
    //     smw.server_send_rpc_call_result();
    //     exit(0);
    // }

    // smw.rpc_id = 3;
    // smw.num_args = 2;
    // smw.args[0] = "RPCARG1";
    // smw.args[1] = "RPCARG2";
    // smw.make_rpc_call();

    // cout << "Recieved rpc result:" << endl << smw.rpc_id << endl << smw.num_args << endl << smw.args[0] << endl;
}

void shared_mem_destroy_test() {
    // Max number of shared memory segments can have in use "at one time" is 4096
    // This is fine, but need to make sure that actually destroying the shared memory
    // Following code should crash if not cleaning up shared memory segments properly in SharedMemWrapper destructor
    // There also seems to be a limit on semaphores, but not been running into that
    vector<shared_ptr<SharedMemWrapper>> smw_vec;
    for (int i=0; i<100; i++) {
        string a_filename = "/";
        smw_vec.push_back(make_shared<SharedMemWrapper>(a_filename,i,1024));
        smw_vec[i].reset();
    }
    for (int j=0; j<50; j++) {
        for (int i=0; i<100; i++) {
            // cout << j*100 + i << endl;
            string a_filename = "/";
            smw_vec[i] = make_shared<SharedMemWrapper>(a_filename,i,1024);
        }
        for (int i=0; i<100; i++) {
            smw_vec[i].reset();
        }
    }
}

/**
 * test pybind gil objects
 */
void _pybind_gil_test_helper_t0(int thread_id) {
    // no gil held
    py::gil_scoped_acquire acquire;
    // gil held
    cout << "In thread id: " << thread_id << ", 0s sleep with gil, (1ack), should appear at 0s" << endl;
    sleep(2);
    {
        cout << "In thread id: " << thread_id << ", 1s sleep w/gil, (1ack,1rel), should appear at 2s" << endl;
        py::gil_scoped_release release;
        //no gil held
        sleep(2);
    }
    // gil held (scoped release out of scope => reacquire)
    cout << "In thread id: " << thread_id << ", 1s sleep w/gil, 2s sleep w/out, (1ack), should appeat at 6s "
        << "(but second at 6s)" << endl;
}
void _pybind_gil_test_helper_t1(int thread_id) {
    // no gil held
    py::gil_scoped_acquire acquire;
    // gil held
    cout << "In thread id: " << thread_id << ", 0s sleep, should appear at 2s (but second at 2s)" << endl;
    sleep(2);
    cout << "In thread id: " << thread_id << ", 2s sleep, should appear at 4s" << endl;
    sleep(2);
    cout << "In thread id: " << thread_id << ", 4s sleep, should appear at 6s" << endl;
}

/**
 * Working out how to use gil with scoped_interpreter
 * 
 * gil is acquired by default, so need to release it in main thread first
 * 
 * Want to check if gil_scoped_acquire releases the gil in its destructor, as it didn't seem clear in docs
 */
void pybind_gil_test() {
    py::gil_scoped_release release;
    thread thread_one(&_pybind_gil_test_helper_t0, 0);
    sleep(1);
    thread thread_two(&_pybind_gil_test_helper_t1, 1);
    thread_one.join();
    thread_two.join();
}

void py_thts_env_test(double alpha, bool use_python_env) {
    // release gil
    py::gil_scoped_release rel;

    // params
    int env_size = 3;
    double stay_prob = 0.1;
    int num_trials = 10000;
    int print_tree_depth = 2;
    int num_threads = 3;

    // Make py env (making a py::object of python thts env, and pass into constructor)
    shared_ptr<ThtsEnv> thts_env;
    if (use_python_env) {
        py::gil_scoped_acquire acq;

        shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
        string thts_unique_filename = "/";

        // py::module_ py_thts_env_module = py::module_::import("test_env"); 
        // py::object py_thts_env = py_thts_env_module.attr("PyTestThtsEnv")(env_size, stay_prob);
        // thts_env = make_shared<PyMultiprocessingThtsEnv>(pickle_wrapper, make_shared<py::object>(py_thts_env));

        // py::module_ py_thts_env_module = py::module_::import("test_env"); 
        // py::dict kw_args;
        // kw_args["grid_size"] = to_string(env_size);
        // kw_args["stay_prob"] = to_string(stay_prob);
        // py::object py_thts_env = py_thts_env_module.attr("PyTestThtsEnv")(**kw_args);
        // thts_env = make_shared<PyMultiprocessingThtsEnv>(pickle_wrapper, make_shared<py::object>(py_thts_env));

        py::dict kw_args;
        kw_args["grid_size"] = to_string(env_size);
        kw_args["stay_prob"] = to_string(stay_prob);
        thts_env = make_shared<PyMultiprocessingThtsEnv>(
            pickle_wrapper, thts_unique_filename, "test_env", "PyTestThtsEnv", make_shared<py::dict>(kw_args));
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
        // py::gil_scoped_acquire acq;

        DentsManagerArgs args(thts_env);
        args.seed = 60415;
        args.max_depth = env_size * 4;
        args.mcts_mode = false;
        args.temp = alpha;
        args.num_threads = num_threads;
        args.num_envs = num_threads;
        manager = make_shared<DentsManager>(args);

        // Setup python servers
        if (use_python_env) {
            for (int i=0; i<args.num_envs; i++) {
                PyMultiprocessingThtsEnv& py_mp_env = *dynamic_pointer_cast<PyMultiprocessingThtsEnv>(manager->thts_env(i));
                py_mp_env.start_python_server(i);
            }
        }
        
        root_node = make_shared<EstDNode>(manager, thts_env->get_initial_state_itfc(), 0, 0);
        bts_pool = make_shared<ThtsPool>(manager, root_node, num_threads);
    }

    // Run thts trials (same as c++)
    // Needs to not have the gil, so threads can grab it any make interpreters
    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();
    bts_pool->run_trials(num_trials);
    std::chrono::duration<double> dur = chrono::system_clock::now() - start_time;

    // Print out a tree (same as c++)
    // Make sure have gil, because getting pretty print string using python objects
    // py::gil_scoped_acquire acq;
    cout << "EST with " << num_threads << " threads (took " << dur.count() << ")";
    if (print_tree_depth > 0) {
        cout << " and looks like:\n";
        cout << root_node->get_pretty_print_string(print_tree_depth) << endl;
    } else {
        cout << endl;
    }
    
    // Keep on crashing after here
    // TODO: sort out for python lib release, because need to not crash at least until Py_Finalize to use in Python code
    // Annoying to run ipcrm -v -a all the time, so make sure thats not necessary by the time it crashes

    // if (use_python_env) {
    //     for (int i=0; i<num_threads; i++) {
    //         PyMultiprocessingThtsEnv& py_mp_env = *dynamic_pointer_cast<PyMultiprocessingThtsEnv>(manager->thts_env(i));
    //         py_mp_env.clear_unix_sem_and_shm();
    //     }
    // }

    // Force tree destructors to be called (to clean up python objects with the gil)
    // resetting smart pointers should refcount to zero and call destructor
    // A bit annoying having to make everything a smart pointer to call destructors with gil, but mech
    // py::gil_scoped_acquire acq;
    // manager.reset();
    // manager_args.reset();
    // thts_env.reset();
    // bts_pool.reset();
    // root_node.reset();
}

void czt_test() {

    // params
    double bias = 4.0;
    int num_backups_before_allowed_to_split = 10;

    int walk_len = 5;
    double stay_prob = 0.0;

    int num_trials = 10000;
    int print_tree_depth = 2;
    int num_threads = 4;

    // Setup env 
    shared_ptr<thts::test::TestMoThtsEnv> thts_env = make_shared<thts::test::TestMoThtsEnv>(walk_len, stay_prob);

    // Make thts manager 
    shared_ptr<CztManagerArgs> args = make_shared<CztManagerArgs>(thts_env);
    args->seed = 60415;
    args->max_depth = walk_len * 4;
    args->mcts_mode = false;
    args->bias = bias;
    args->num_backups_before_allowed_to_split = num_backups_before_allowed_to_split;
    args->num_threads = num_threads;
    args->num_envs = num_threads; 
    shared_ptr<CztManager> manager = make_shared<CztManager>(*args);
 
    // // Setup python servers
    // for (int i=0; i<args.num_envs; i++) {
    //     PyMultiprocessingThtsEnv& py_mp_env = (PyMultiprocessingThtsEnv&) *manager->thts_env(i);
    //     py_mp_env.start_python_server(i);
    // }

    // Run search and time, remembering to unlock the python gil if we have one, so subthreads can grab GIL to make 
    // subinterpreters
    shared_ptr<const State> init_state = thts_env->get_initial_state_itfc();
    shared_ptr<CztDNode> root_node = make_shared<CztDNode>(manager, init_state, 0, 0);
    // shared_ptr<ThtsPool> thts_pool = make_shared<PyThtsPool>(manager, root_node, num_threads);
    shared_ptr<ThtsPool> thts_pool = make_shared<MoThtsPool>(manager, root_node, num_threads);
    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();
    // py::gil_scoped_release rel;
    thts_pool->run_trials(num_trials);
    std::chrono::duration<double> dur = chrono::system_clock::now() - start_time;

    // Print out a tree
    // Make sure have gil again if using python objects
    // py::gil_scoped_acquire acq;
    cout << "CZT with " << num_threads << " threads (took " << dur.count() << ")";
    if (print_tree_depth > 0) {
        cout << " and looks like:\n";
        cout << root_node->get_pretty_print_string(print_tree_depth);
    } 
    cout << endl << endl; 

    // Pretty ball lists
    cout << "Printing CZT ball lists for first decision." << endl << endl;
    ThtsEnvContext ctx;
    shared_ptr<ActionVector> actions = thts_env->get_valid_actions_itfc(init_state,ctx);
    for (shared_ptr<const Action> action : *actions) {
        cout << "CZT ball list for action " << *action << ":" << endl;
        cout << root_node->get_child_node(action)->get_ball_list_pretty_print_string() << endl << endl;
    }
    
    // Test out Mo MC Eval
    int num_eval_rollouts = 250;
    shared_ptr<EvalPolicy> policy = make_shared<EvalPolicy>(root_node, thts_env, manager);
    MoMCEvaluator mo_mc_eval(
        policy,  
        manager->max_depth,
        manager,
        Eigen::ArrayXd::Zero(2)-walk_len,
        Eigen::ArrayXd::Zero(2)-0.5*walk_len);
    // py::gil_scoped_release rel2;
    mo_mc_eval.run_rollouts(num_eval_rollouts, num_threads);

    cout << "CZT evaluations from MoMCEval." << endl;
    cout << "Mean MO return." << endl;
    cout << mo_mc_eval.get_mean_mo_return() << endl;
    cout << "Mean MO ctx return." << endl;
    cout << mo_mc_eval.get_mean_mo_ctx_return() << endl;
    cout << "Mean MO normalised ctx return." << endl;
    cout << mo_mc_eval.get_mean_mo_normalised_ctx_return() << endl;

    // // Trying to make python embedding exit gracefully stuff (see py_thts_env_test to understand)
    // for (int i=0; i<num_threads; i++) {
    //     PyMultiprocessingThtsEnv& py_mp_env = (PyMultiprocessingThtsEnv&) *manager->thts_env(i);
    //     py_mp_env.clear_unix_sem_and_shm();
    // }
    // manager.reset();
    // manager_args.reset();
    // thts_env.reset();
    // thts_pool.reset();
    // root_node.reset();
}

void czt_4d_test() {

    // params
    double bias = 4.0;
    int num_backups_before_allowed_to_split = 10;

    int walk_len = 5;
    double stay_prob = 0.0;

    int num_trials = 10000;
    int print_tree_depth = 2;
    int num_threads = 4;

    // Setup env 
    shared_ptr<thts::test::TestMoThtsEnv> thts_env = make_shared<thts::test::TestMoThtsEnv>(walk_len, stay_prob, true);

    // Make thts manager 
    shared_ptr<CztManagerArgs> args = make_shared<CztManagerArgs>(thts_env);
    args->seed = 60415;
    args->max_depth = walk_len * 4;
    args->mcts_mode = false;
    args->bias = bias;
    args->num_backups_before_allowed_to_split = num_backups_before_allowed_to_split;
    args->num_threads = num_threads;
    args->num_envs = num_threads; 
    shared_ptr<CztManager> manager = make_shared<CztManager>(*args);
 
    // // Setup python servers
    // for (int i=0; i<args.num_envs; i++) {
    //     PyMultiprocessingThtsEnv& py_mp_env = (PyMultiprocessingThtsEnv&) *manager->thts_env(i);
    //     py_mp_env.start_python_server(i);
    // }

    // Run search and time, remembering to unlock the python gil if we have one, so subthreads can grab GIL to make 
    // subinterpreters
    shared_ptr<const State> init_state = thts_env->get_initial_state_itfc();
    shared_ptr<CztDNode> root_node = make_shared<CztDNode>(manager, init_state, 0, 0);
    // shared_ptr<ThtsPool> thts_pool = make_shared<PyThtsPool>(manager, root_node, num_threads);
    shared_ptr<ThtsPool> thts_pool = make_shared<MoThtsPool>(manager, root_node, num_threads);
    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();
    // py::gil_scoped_release rel;
    thts_pool->run_trials(num_trials);
    std::chrono::duration<double> dur = chrono::system_clock::now() - start_time;

    // Print out a tree
    // Make sure have gil again if using python objects
    // py::gil_scoped_acquire acq;
    cout << "CZT with " << num_threads << " threads (took " << dur.count() << ")";
    if (print_tree_depth > 0) {
        cout << " and looks like:\n";
        cout << root_node->get_pretty_print_string(print_tree_depth);
    } 
    cout << endl << endl; 

    // Pretty ball lists
    cout << "Printing CZT ball lists for first decision." << endl << endl;
    ThtsEnvContext ctx;
    shared_ptr<ActionVector> actions = thts_env->get_valid_actions_itfc(init_state,ctx);
    for (shared_ptr<const Action> action : *actions) {
        cout << "CZT ball list for action " << *action << ":" << endl;
        cout << root_node->get_child_node(action)->get_ball_list_pretty_print_string() << endl << endl;
    }
    
    // Test out Mo MC Eval
    int num_eval_rollouts = 250;
    shared_ptr<EvalPolicy> policy = make_shared<EvalPolicy>(root_node, thts_env, manager);
    MoMCEvaluator mo_mc_eval(
        policy,  
        manager->max_depth,
        manager,
        Eigen::ArrayXd::Zero(4)-walk_len,
        Eigen::ArrayXd::Ones(4)/(1.0-thts_env->get_gamma()));
    // py::gil_scoped_release rel2;
    mo_mc_eval.run_rollouts(num_eval_rollouts, num_threads);

    cout << "CZT evaluations from MoMCEval." << endl;
    cout << "Mean MO return." << endl;
    cout << mo_mc_eval.get_mean_mo_return() << endl;
    cout << "Mean MO ctx return." << endl;
    cout << mo_mc_eval.get_mean_mo_ctx_return() << endl;
    cout << "Mean MO normalised ctx return." << endl;
    cout << mo_mc_eval.get_mean_mo_normalised_ctx_return() << endl;

    // // Trying to make python embedding exit gracefully stuff (see py_thts_env_test to understand)
    // for (int i=0; i<num_threads; i++) {
    //     PyMultiprocessingThtsEnv& py_mp_env = (PyMultiprocessingThtsEnv&) *manager->thts_env(i);
    //     py_mp_env.clear_unix_sem_and_shm();
    // }
    // manager.reset();
    // manager_args.reset();
    // thts_env.reset();
    // thts_pool.reset();
    // root_node.reset();
}

void gym_env_test() {
    py::gil_scoped_release release;

    // params
    string gym_env_id = "FrozenLake-v1";
    int num_trials = 100;
    int print_tree_depth = 2;
    int num_threads = 4;
    double alpha = 1.0;

    // Make py env (making a py::object of python thts env, and pass into constructor)
    shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
    string thts_unique_filename = "/";
    shared_ptr<ThtsEnv> thts_env = make_shared<GymMultiprocessingThtsEnv>(
        pickle_wrapper, thts_unique_filename, gym_env_id);

    // Make thts manager with the py env (same as c++ (use unit tests))
    // But protect with GIL for any python ops in creating things
    DentsManagerArgs args(thts_env);
    args.seed = 60415;
    args.max_depth = 25;
    args.mcts_mode = false;
    args.temp = alpha;
    args.num_threads = num_threads;
    args.num_envs = num_threads;
    shared_ptr<DentsManager> manager = make_shared<DentsManager>(args);

    // Setup python servers
    for (int i=0; i<args.num_envs; i++) {
        PyMultiprocessingThtsEnv& py_mp_env = *dynamic_pointer_cast<PyMultiprocessingThtsEnv>(manager->thts_env(i));
        py_mp_env.start_python_server(i);
    }

    shared_ptr<EstDNode> root_node = make_shared<EstDNode>(manager, thts_env->get_initial_state_itfc(), 0, 0);
    shared_ptr<ThtsPool> thts_pool = make_shared<ThtsPool>(manager, root_node, num_threads);

    // Run thts trials (same as c++)
    // Needs to not have the gil, so threads can grab it any make interpreters
    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();
    thts_pool->run_trials(num_trials);
    std::chrono::duration<double> dur = chrono::system_clock::now() - start_time;

    // Print out a tree (same as c++)
    // Make sure have gil, because getting pretty print string using python objects
    cout << "EST with " << num_threads << " threads (took " << dur.count() << ")";
    if (print_tree_depth > 0) {
        cout << " and looks like:\n";
        cout << root_node->get_pretty_print_string(print_tree_depth) << endl;
    } else {
        cout << endl;
    }

    // Keep on crashing after here
    // TODO: sort out for python lib release, because need to not crash at least until Py_Finalize to use in Python code
    // Annoying to run ipcrm -v -a all the time, so make sure thats not necessary by the time it crashes
    // for (int i=0; i<num_threads; i++) {
    //     PyMultiprocessingThtsEnv& py_mp_env = *dynamic_pointer_cast<PyMultiprocessingThtsEnv>(manager->thts_env(i));
    //     py_mp_env.clear_unix_sem_and_shm();
    // }

    // Force tree destructors to be called (to clean up python objects with the gil)
    // resetting smart pointers should refcount to zero and call destructor
    // A bit annoying having to make everything a smart pointer to call destructors with gil, but mech
    // manager.reset();
    // thts_env.reset();
    // thts_pool.reset();
    // root_node.reset();
}

void mo_gym_env_test(string thts_unique_filename="/") {
    py::gil_scoped_release release;

    // params
    double bias = 4.0;
    int num_backups_before_allowed_to_split = 10;

    int num_trials = 10000;
    int print_tree_depth = 2;
    int num_threads = 4;

    // Setup env 
    string mo_gym_env_id = "deep-sea-treasure-v0";
    shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
    shared_ptr<MoGymMultiprocessingThtsEnv> thts_env = make_shared<MoGymMultiprocessingThtsEnv>(
        pickle_wrapper, thts_unique_filename, mo_gym_env_id);

    // Make thts manager 
    shared_ptr<CztManagerArgs> args = make_shared<CztManagerArgs>(thts_env);
    args->seed = 60415;
    args->max_depth = 25;
    args->mcts_mode = false;
    args->bias = bias;
    args->num_backups_before_allowed_to_split = num_backups_before_allowed_to_split;
    args->num_threads = num_threads;
    args->num_envs = num_threads;
    shared_ptr<CztManager> manager = make_shared<CztManager>(*args);

    // Setup python servers
    for (int i=0; i<args->num_envs; i++) {
        PyMultiprocessingThtsEnv& py_mp_env = *dynamic_pointer_cast<PyMultiprocessingThtsEnv>(manager->thts_env(i));
        py_mp_env.start_python_server(i);
    }

    // Run search and time, remembering to unlock the python gil if we have one, so subthreads can grab GIL to make 
    // subinterpreters
    shared_ptr<const State> init_state = thts_env->get_initial_state_itfc();
    shared_ptr<CztDNode> root_node = make_shared<CztDNode>(manager, init_state, 0, 0);
    shared_ptr<ThtsPool> thts_pool = make_shared<MoThtsPool>(manager, root_node, num_threads);
    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();
    thts_pool->run_trials(num_trials);
    thts_pool->run_trials(num_trials);
    std::chrono::duration<double> dur = chrono::system_clock::now() - start_time;

    // Print out a tree
    // Make sure have gil again if using python objects
    cout << "CZT with " << num_threads << " threads (took " << dur.count() << ")";
    if (print_tree_depth > 0) {
        cout << " and looks like:\n";
        cout << root_node->get_pretty_print_string(print_tree_depth);
    }
    cout << endl << endl; 

    // Pretty ball lists
    cout << "Printing CZT ball lists for first decision." << endl << endl;
    ThtsEnvContext ctx;
    shared_ptr<ActionVector> actions = thts_env->get_valid_actions_itfc(init_state,ctx);
    for (shared_ptr<const Action> action : *actions) {
        cout << "CZT ball list for action " << *action << ":" << endl;
        cout << root_node->get_child_node(action)->get_ball_list_pretty_print_string() << endl << endl;
    }
    
    // Test out Mo MC Eval 
    int num_eval_rollouts = 1000;
    shared_ptr<EvalPolicy> policy = make_shared<EvalPolicy>(root_node, thts_env, manager);
    Eigen::ArrayXd r_min(2);
    r_min[0] = 0.0;
    r_min[1] = -1.0 * manager->max_depth;
    Eigen::ArrayXd r_max(2);
    r_max[0] = 23.7;
    r_max[1] = 0.0;
    MoMCEvaluator mo_mc_eval(
        policy, 
        manager->max_depth,
        manager,
        r_min,
        r_max);
    mo_mc_eval.run_rollouts(num_eval_rollouts, num_threads);

    cout << "CZT evaluations from MoMCEval." << endl;
    cout << "Mean MO return." << endl;
    cout << mo_mc_eval.get_mean_mo_return() << endl;
    cout << "Mean MO ctx return." << endl;
    cout << mo_mc_eval.get_mean_mo_ctx_return() << endl;
    cout << "Mean MO normalised ctx return." << endl;
    cout << mo_mc_eval.get_mean_mo_normalised_ctx_return() << endl;

    // Trying to make python embedding exit gracefully stuff (see py_thts_env_test to understand)
    // for (int i=0; i<num_threads; i++) {
    //     PyMultiprocessingThtsEnv& py_mp_env = *dynamic_pointer_cast<PyMultiprocessingThtsEnv>(manager->thts_env(i));
    //     py_mp_env.clear_unix_sem_and_shm();
    // }
    // manager.reset();
    // args.reset();
    // thts_env.reset();
    // thts_pool.reset();
    // root_node.reset();
}

void eigen_svd_test() {
    // Eigen is really annoting sometimes... 
    // 3 points
    // Eigen::VectorXd v0 {1.0, 1.0, 1.0};
    // Eigen::VectorXd v1 {1.0, 2.0, 1.0};
    // Eigen::VectorXd v2 {2.0, 1.0, 1.0};

    // // Make matrix of vectors in the plane (want to compute a normal)
    // Eigen::MatrixXd mat(3,2);
    // mat.col(0) = v1 - v0;
    // mat.col(1) = v2 - v0;

    Eigen::MatrixXd mat 
    {
        {0.0, 1.0},
        {1.0, 0.0},
        {0.0, 0.0},
    };

    cout << "mat is:" << endl << mat << endl;
    cout << "mat.col(0) is:" << endl << mat.col(0) << endl;
    cout << "mat(0,0) is:" << endl << mat(0,0) << endl;

    // Compute SVD
    Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::ComputeFullU | Eigen::ComputeThinV> svd(mat);

    cout << "svd.singularValues() is:" << endl << svd.singularValues() << endl;
    cout << "svd.matrixU() is:" << endl << svd.matrixU() << endl;
    cout << "svd.matrixV() is:" << endl << svd.matrixV() << endl;

    // Check that we can cast back to an array
    Eigen::ArrayXd normal = svd.matrixU().col(2).array();
    cout << "Normal vector as an array is:" << endl << normal << endl;
}

void sm_bts_test() {

    // params
    int walk_len = 5;
    double stay_prob = 0.0;

    int num_trials = 10000;
    int print_tree_depth = 2;
    int num_threads = 4;

    // Setup env 
    shared_ptr<thts::test::TestMoThtsEnv> thts_env = make_shared<thts::test::TestMoThtsEnv>(walk_len, stay_prob);

    // Make thts manager 
    Eigen::ArrayXd default_val = Eigen::ArrayXd::Zero(2) - walk_len * 4;
    shared_ptr<SmtBtsManagerArgs> args = make_shared<SmtBtsManagerArgs>(thts_env, default_val);
    args->seed = 60415;
    args->max_depth = walk_len * 4;
    args->mcts_mode = false;
    args->num_threads = num_threads;
    args->num_envs = num_threads; 
    args->simplex_map_splitting_option = SPLIT_triangulation;
    shared_ptr<SmtBtsManager> manager = make_shared<SmtBtsManager>(*args);
 
    // // Setup python servers
    // for (int i=0; i<args.num_envs; i++) {
    //     PyMultiprocessingThtsEnv& py_mp_env = (PyMultiprocessingThtsEnv&) *manager->thts_env(i);
    //     py_mp_env.start_python_server(i);
    // }

    // Run search and time, remembering to unlock the python gil if we have one, so subthreads can grab GIL to make 
    // subinterpreters
    shared_ptr<const State> init_state = thts_env->get_initial_state_itfc();
    shared_ptr<SmtBtsDNode> root_node = make_shared<SmtBtsDNode>(manager, init_state, 0, 0);
    // shared_ptr<ThtsPool> thts_pool = make_shared<PyThtsPool>(manager, root_node, num_threads);
    shared_ptr<ThtsPool> thts_pool = make_shared<MoThtsPool>(manager, root_node, num_threads);
    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();
    // py::gil_scoped_release rel;
    thts_pool->run_trials(num_trials);
    std::chrono::duration<double> dur = chrono::system_clock::now() - start_time;

    // Print out a tree
    // Make sure have gil again if using python objects
    // py::gil_scoped_acquire acq;
    cout << "SM-BTS with " << num_threads << " threads (took " << dur.count() << ")";
    if (print_tree_depth > 0) {
        cout << " and looks like:\n";
        cout << root_node->get_pretty_print_string(print_tree_depth);
    } 
    cout << endl << endl; 

    // Pretty ball lists
    cout << "Printing SM-BTS simplex map at root node ball lists for first decision." << endl;
    cout << root_node->get_simplex_map_pretty_print_string() << endl << endl;
    ThtsEnvContext ctx;
    shared_ptr<ActionVector> actions = thts_env->get_valid_actions_itfc(init_state,ctx);
    for (shared_ptr<const Action> action : *actions) {
        cout << "Simplex map ball list for action " << *action << ":" << endl;
        cout << root_node->get_child_node(action)->get_simplex_map_pretty_print_string() << endl << endl;
    }
    
    // Test out Mo MC Eval
    int num_eval_rollouts = 250;
    shared_ptr<EvalPolicy> policy = make_shared<EvalPolicy>(root_node, thts_env, manager);
    MoMCEvaluator mo_mc_eval(
        policy,  
        manager->max_depth,
        manager,
        Eigen::ArrayXd::Zero(2)-walk_len,
        Eigen::ArrayXd::Zero(2)-0.5*walk_len); 
    // py::gil_scoped_release rel2;
    mo_mc_eval.run_rollouts(num_eval_rollouts, num_threads);

    cout << "SM-BTS evaluations from MoMCEval." << endl;
    cout << "Mean MO return." << endl;
    cout << mo_mc_eval.get_mean_mo_return() << endl;
    cout << "Mean MO ctx return." << endl;
    cout << mo_mc_eval.get_mean_mo_ctx_return() << endl;
    cout << "Mean MO normalised ctx return." << endl;
    cout << mo_mc_eval.get_mean_mo_normalised_ctx_return() << endl;

    // // Trying to make python embedding exit gracefully stuff (see py_thts_env_test to understand)
    // for (int i=0; i<num_threads; i++) {
    //     PyMultiprocessingThtsEnv& py_mp_env = (PyMultiprocessingThtsEnv&) *manager->thts_env(i);
    //     py_mp_env.clear_unix_sem_and_shm();
    // }
    // manager.reset();
    // manager_args.reset();
    // thts_env.reset();
    // thts_pool.reset();
    // root_node.reset();
}

void sm_bts_4d_test() {

    // params
    int walk_len = 5;
    double stay_prob = 0.0;

    int num_trials = 10000;
    int print_tree_depth = 2;
    int num_threads = 4;

    // Setup env 
    shared_ptr<thts::test::TestMoThtsEnv> thts_env = make_shared<thts::test::TestMoThtsEnv>(walk_len, stay_prob, true);

    // Make thts manager 
    Eigen::ArrayXd default_val = Eigen::ArrayXd::Zero(4) - walk_len * 4;
    shared_ptr<SmtBtsManagerArgs> args = make_shared<SmtBtsManagerArgs>(thts_env, default_val);
    // args->seed = 60415;
    args->max_depth = walk_len * 4;
    args->mcts_mode = false;
    args->num_threads = num_threads;
    args->num_envs = num_threads; 
    args->simplex_node_max_depth = 3;
    args->simplex_map_splitting_option = SPLIT_triangulation;
    shared_ptr<SmtBtsManager> manager = make_shared<SmtBtsManager>(*args);
 
    // // Setup python servers
    // for (int i=0; i<args.num_envs; i++) {
    //     PyMultiprocessingThtsEnv& py_mp_env = (PyMultiprocessingThtsEnv&) *manager->thts_env(i);
    //     py_mp_env.start_python_server(i);
    // }

    // Run search and time, remembering to unlock the python gil if we have one, so subthreads can grab GIL to make 
    // subinterpreters
    shared_ptr<const State> init_state = thts_env->get_initial_state_itfc();
    shared_ptr<SmtBtsDNode> root_node = make_shared<SmtBtsDNode>(manager, init_state, 0, 0);
    // shared_ptr<ThtsPool> thts_pool = make_shared<PyThtsPool>(manager, root_node, num_threads);
    shared_ptr<ThtsPool> thts_pool = make_shared<MoThtsPool>(manager, root_node, num_threads);
    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();
    // py::gil_scoped_release rel;
    thts_pool->run_trials(num_trials);
    std::chrono::duration<double> dur = chrono::system_clock::now() - start_time;

    // Print out a tree
    // Make sure have gil again if using python objects
    // py::gil_scoped_acquire acq;
    cout << "SM-BTS with " << num_threads << " threads (took " << dur.count() << ")";
    if (print_tree_depth > 0) {
        cout << " and looks like:\n";
        cout << root_node->get_pretty_print_string(print_tree_depth);
    } 
    cout << endl << endl; 

    // Pretty ball lists
    cout << "Printing SM-BTS simplex map at root node ball lists for first decision." << endl;
    cout << root_node->get_simplex_map_pretty_print_string() << endl << endl;
    // ThtsEnvContext ctx;
    // shared_ptr<ActionVector> actions = thts_env->get_valid_actions_itfc(init_state,ctx);
    // for (shared_ptr<const Action> action : *actions) {
    //     cout << "Simplex map ball list for action " << *action << ":" << endl;
    //     cout << root_node->get_child_node(action)->get_simplex_map_pretty_print_string() << endl << endl;
    // }
    
    // Test out Mo MC Eval
    int num_eval_rollouts = 250;
    shared_ptr<EvalPolicy> policy = make_shared<EvalPolicy>(root_node, thts_env, manager);
    MoMCEvaluator mo_mc_eval(
        policy,  
        manager->max_depth,
        manager,
        Eigen::ArrayXd::Zero(4)-walk_len,
        Eigen::ArrayXd::Ones(4)/(1.0-thts_env->get_gamma())); 
    // py::gil_scoped_release rel2;
    mo_mc_eval.run_rollouts(num_eval_rollouts, num_threads);

    cout << "SM-BTS evaluations from MoMCEval." << endl;
    cout << "Mean MO return." << endl;
    cout << mo_mc_eval.get_mean_mo_return() << endl;
    cout << "Mean MO ctx return." << endl;
    cout << mo_mc_eval.get_mean_mo_ctx_return() << endl;
    cout << "Mean MO normalised ctx return." << endl;
    cout << mo_mc_eval.get_mean_mo_normalised_ctx_return() << endl;

    // // Trying to make python embedding exit gracefully stuff (see py_thts_env_test to understand)
    // for (int i=0; i<num_threads; i++) {
    //     PyMultiprocessingThtsEnv& py_mp_env = (PyMultiprocessingThtsEnv&) *manager->thts_env(i);
    //     py_mp_env.clear_unix_sem_and_shm();
    // }
    // manager.reset();
    // manager_args.reset();
    // thts_env.reset();
    // thts_pool.reset();
    // root_node.reset();
}

void sm_dents_test() {

    // params
    int walk_len = 5;
    double stay_prob = 0.0;

    int num_trials = 10000;
    int print_tree_depth = 2;
    int num_threads = 4;

    // Setup env 
    shared_ptr<thts::test::TestMoThtsEnv> thts_env = make_shared<thts::test::TestMoThtsEnv>(walk_len, stay_prob);

    // Make thts manager 
    Eigen::ArrayXd default_val = Eigen::ArrayXd::Zero(2) - walk_len * 4;
    shared_ptr<SmtDentsManagerArgs> args = make_shared<SmtDentsManagerArgs>(thts_env, default_val);
    args->seed = 60415;
    args->max_depth = walk_len * 4;
    args->mcts_mode = false;
    args->num_threads = num_threads;
    args->num_envs = num_threads; 
    shared_ptr<SmtDentsManager> manager = make_shared<SmtDentsManager>(*args);
 
    // // Setup python servers
    // for (int i=0; i<args.num_envs; i++) {
    //     PyMultiprocessingThtsEnv& py_mp_env = (PyMultiprocessingThtsEnv&) *manager->thts_env(i);
    //     py_mp_env.start_python_server(i);
    // }

    // Run search and time, remembering to unlock the python gil if we have one, so subthreads can grab GIL to make 
    // subinterpreters
    shared_ptr<const State> init_state = thts_env->get_initial_state_itfc();
    shared_ptr<SmtDentsDNode> root_node = make_shared<SmtDentsDNode>(manager, init_state, 0, 0);
    // shared_ptr<ThtsPool> thts_pool = make_shared<PyThtsPool>(manager, root_node, num_threads);
    shared_ptr<ThtsPool> thts_pool = make_shared<MoThtsPool>(manager, root_node, num_threads);
    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();
    // py::gil_scoped_release rel;
    thts_pool->run_trials(num_trials);
    std::chrono::duration<double> dur = chrono::system_clock::now() - start_time;

    // Print out a tree
    // Make sure have gil again if using python objects
    // py::gil_scoped_acquire acq;
    cout << "SM-DENTS with " << num_threads << " threads (took " << dur.count() << ")";
    if (print_tree_depth > 0) {
        cout << " and looks like:\n";
        cout << root_node->get_pretty_print_string(print_tree_depth);
    } 
    cout << endl << endl; 

    // Pretty ball lists
    cout << "Printing SM-DENTS simplex map at root node ball lists for first decision." << endl;
    cout << root_node->get_simplex_map_pretty_print_string() << endl << endl;
    ThtsEnvContext ctx;
    shared_ptr<ActionVector> actions = thts_env->get_valid_actions_itfc(init_state,ctx);
    for (shared_ptr<const Action> action : *actions) {
        cout << "Simplex map ball list for action " << *action << ":" << endl;
        cout << root_node->get_child_node(action)->get_simplex_map_pretty_print_string() << endl << endl;
    }
    
    // Test out Mo MC Eval
    int num_eval_rollouts = 250;
    shared_ptr<EvalPolicy> policy = make_shared<EvalPolicy>(root_node, thts_env, manager);
    MoMCEvaluator mo_mc_eval(
        policy,  
        manager->max_depth,
        manager,
        Eigen::ArrayXd::Zero(2)-walk_len,
        Eigen::ArrayXd::Zero(2)-0.5*walk_len); 
    // py::gil_scoped_release rel2;
    mo_mc_eval.run_rollouts(num_eval_rollouts, num_threads);

    cout << "SM-DENTS evaluations from MoMCEval." << endl;
    cout << "Mean MO return." << endl;
    cout << mo_mc_eval.get_mean_mo_return() << endl;
    cout << "Mean MO ctx return." << endl;
    cout << mo_mc_eval.get_mean_mo_ctx_return() << endl;
    cout << "Mean MO normalised ctx return." << endl;
    cout << mo_mc_eval.get_mean_mo_normalised_ctx_return() << endl;

    // // Trying to make python embedding exit gracefully stuff (see py_thts_env_test to understand)
    // for (int i=0; i<num_threads; i++) {
    //     PyMultiprocessingThtsEnv& py_mp_env = (PyMultiprocessingThtsEnv&) *manager->thts_env(i);
    //     py_mp_env.clear_unix_sem_and_shm();
    // }
    // manager.reset();
    // manager_args.reset();
    // thts_env.reset();
    // thts_pool.reset();
    // root_node.reset();
}

void sm_bts_bin_tree_test() {

    // params
    int walk_len = 5;
    double stay_prob = 0.0;

    int num_trials = 10000;
    int print_tree_depth = 2;
    int num_threads = 4;

    // Setup env 
    shared_ptr<thts::test::TestMoThtsEnv> thts_env = make_shared<thts::test::TestMoThtsEnv>(walk_len, stay_prob);

    // Make thts manager 
    Eigen::ArrayXd default_val = Eigen::ArrayXd::Zero(2) - walk_len * 4;
    shared_ptr<SmtBtsManagerArgs> args = make_shared<SmtBtsManagerArgs>(thts_env, default_val);
    args->seed = 60415;
    args->max_depth = walk_len * 4;
    args->mcts_mode = false;
    args->num_threads = num_threads;
    args->num_envs = num_threads; 
    shared_ptr<SmtBtsManager> manager = make_shared<SmtBtsManager>(*args);
 
    // // Setup python servers
    // for (int i=0; i<args.num_envs; i++) {
    //     PyMultiprocessingThtsEnv& py_mp_env = (PyMultiprocessingThtsEnv&) *manager->thts_env(i);
    //     py_mp_env.start_python_server(i);
    // }

    // Run search and time, remembering to unlock the python gil if we have one, so subthreads can grab GIL to make 
    // subinterpreters
    shared_ptr<const State> init_state = thts_env->get_initial_state_itfc();
    shared_ptr<SmtBtsDNode> root_node = make_shared<SmtBtsDNode>(manager, init_state, 0, 0);
    // shared_ptr<ThtsPool> thts_pool = make_shared<PyThtsPool>(manager, root_node, num_threads);
    shared_ptr<ThtsPool> thts_pool = make_shared<MoThtsPool>(manager, root_node, num_threads);
    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();
    // py::gil_scoped_release rel;
    thts_pool->run_trials(num_trials);
    std::chrono::duration<double> dur = chrono::system_clock::now() - start_time;

    // Print out a tree
    // Make sure have gil again if using python objects
    // py::gil_scoped_acquire acq;
    cout << "SM-BTS with " << num_threads << " threads (took " << dur.count() << ")";
    if (print_tree_depth > 0) {
        cout << " and looks like:\n";
        cout << root_node->get_pretty_print_string(print_tree_depth);
    } 
    cout << endl << endl; 

    // Pretty ball lists
    cout << "Printing SM-BTS simplex map at root node ball lists for first decision." << endl;
    cout << root_node->get_simplex_map_pretty_print_string() << endl << endl;
    ThtsEnvContext ctx;
    shared_ptr<ActionVector> actions = thts_env->get_valid_actions_itfc(init_state,ctx);
    for (shared_ptr<const Action> action : *actions) {
        cout << "Simplex map ball list for action " << *action << ":" << endl;
        cout << root_node->get_child_node(action)->get_simplex_map_pretty_print_string() << endl << endl;
    }
    
    // Test out Mo MC Eval
    int num_eval_rollouts = 250;
    shared_ptr<EvalPolicy> policy = make_shared<EvalPolicy>(root_node, thts_env, manager);
    MoMCEvaluator mo_mc_eval(
        policy,  
        manager->max_depth,
        manager,
        Eigen::ArrayXd::Zero(2)-walk_len,
        Eigen::ArrayXd::Zero(2)-0.5*walk_len); 
    // py::gil_scoped_release rel2;
    mo_mc_eval.run_rollouts(num_eval_rollouts, num_threads);

    cout << "SM-BTS evaluations from MoMCEval." << endl;
    cout << "Mean MO return." << endl;
    cout << mo_mc_eval.get_mean_mo_return() << endl;
    cout << "Mean MO ctx return." << endl;
    cout << mo_mc_eval.get_mean_mo_ctx_return() << endl;
    cout << "Mean MO normalised ctx return." << endl;
    cout << mo_mc_eval.get_mean_mo_normalised_ctx_return() << endl;

    // // Trying to make python embedding exit gracefully stuff (see py_thts_env_test to understand)
    // for (int i=0; i<num_threads; i++) {
    //     PyMultiprocessingThtsEnv& py_mp_env = (PyMultiprocessingThtsEnv&) *manager->thts_env(i);
    //     py_mp_env.clear_unix_sem_and_shm();
    // }
    // manager.reset();
    // manager_args.reset();
    // thts_env.reset();
    // thts_pool.reset();
    // root_node.reset();
}

void sm_bts_bin_tree_4d_test() {

    // params
    int walk_len = 5;
    double stay_prob = 0.0;

    int num_trials = 10000;
    int print_tree_depth = 2;
    int num_threads = 4;

    // Setup env 
    shared_ptr<thts::test::TestMoThtsEnv> thts_env = make_shared<thts::test::TestMoThtsEnv>(walk_len, stay_prob, true);

    // Make thts manager 
    Eigen::ArrayXd default_val = Eigen::ArrayXd::Zero(4) - walk_len * 4;
    shared_ptr<SmtBtsManagerArgs> args = make_shared<SmtBtsManagerArgs>(thts_env, default_val);
    // args->seed = 60415;
    args->max_depth = walk_len * 4;
    args->mcts_mode = false;
    args->num_threads = num_threads;
    args->num_envs = num_threads; 
    args->simplex_node_max_depth = 40;
    shared_ptr<SmtBtsManager> manager = make_shared<SmtBtsManager>(*args);
 
    // // Setup python servers
    // for (int i=0; i<args.num_envs; i++) {
    //     PyMultiprocessingThtsEnv& py_mp_env = (PyMultiprocessingThtsEnv&) *manager->thts_env(i);
    //     py_mp_env.start_python_server(i);
    // }

    // Run search and time, remembering to unlock the python gil if we have one, so subthreads can grab GIL to make 
    // subinterpreters
    shared_ptr<const State> init_state = thts_env->get_initial_state_itfc();
    shared_ptr<SmtBtsDNode> root_node = make_shared<SmtBtsDNode>(manager, init_state, 0, 0);
    // shared_ptr<ThtsPool> thts_pool = make_shared<PyThtsPool>(manager, root_node, num_threads);
    shared_ptr<ThtsPool> thts_pool = make_shared<MoThtsPool>(manager, root_node, num_threads);
    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();
    // py::gil_scoped_release rel;
    thts_pool->run_trials(num_trials);
    std::chrono::duration<double> dur = chrono::system_clock::now() - start_time;

    // Print out a tree
    // Make sure have gil again if using python objects
    // py::gil_scoped_acquire acq;
    cout << "SM-BTS with " << num_threads << " threads (took " << dur.count() << ")";
    if (print_tree_depth > 0) {
        cout << " and looks like:\n";
        cout << root_node->get_pretty_print_string(print_tree_depth);
    } 
    cout << endl << endl; 

    // // Pretty ball lists
    // cout << "Printing SM-BTS simplex map at root node ball lists for first decision." << endl;
    // cout << root_node->get_simplex_map_pretty_print_string() << endl << endl;
    // ThtsEnvContext ctx;
    // shared_ptr<ActionVector> actions = thts_env->get_valid_actions_itfc(init_state,ctx);
    // for (shared_ptr<const Action> action : *actions) {
    //     cout << "Simplex map ball list for action " << *action << ":" << endl;
    //     cout << root_node->get_child_node(action)->get_simplex_map_pretty_print_string() << endl << endl;
    // }
    
    // Test out Mo MC Eval
    int num_eval_rollouts = 250;
    shared_ptr<EvalPolicy> policy = make_shared<EvalPolicy>(root_node, thts_env, manager);
    MoMCEvaluator mo_mc_eval(
        policy,  
        manager->max_depth,
        manager,
        Eigen::ArrayXd::Zero(4)-walk_len,
        Eigen::ArrayXd::Ones(4)/(1.0-thts_env->get_gamma())); 
    // py::gil_scoped_release rel2;
    mo_mc_eval.run_rollouts(num_eval_rollouts, num_threads);

    cout << "SM-BTS evaluations from MoMCEval." << endl;
    cout << "Mean MO return." << endl;
    cout << mo_mc_eval.get_mean_mo_return() << endl;
    cout << "Mean MO ctx return." << endl;
    cout << mo_mc_eval.get_mean_mo_ctx_return() << endl;
    cout << "Mean MO normalised ctx return." << endl;
    cout << mo_mc_eval.get_mean_mo_normalised_ctx_return() << endl;

    // // Trying to make python embedding exit gracefully stuff (see py_thts_env_test to understand)
    // for (int i=0; i<num_threads; i++) {
    //     PyMultiprocessingThtsEnv& py_mp_env = (PyMultiprocessingThtsEnv&) *manager->thts_env(i);
    //     py_mp_env.clear_unix_sem_and_shm();
    // }
    // manager.reset();
    // manager_args.reset();
    // thts_env.reset();
    // thts_pool.reset();
    // root_node.reset();
}

void chmcts_test() {

    // params
    double bias = 4.0;
    int num_backups_before_allowed_to_split = 10;

    int walk_len = 5;
    double stay_prob = 0.0;

    int num_trials = 10000;
    int print_tree_depth = 2;
    int num_threads = 4;

    // Setup env 
    shared_ptr<thts::test::TestMoThtsEnv> thts_env = make_shared<thts::test::TestMoThtsEnv>(walk_len, stay_prob);

    // Make thts manager 
    shared_ptr<ChmctsManagerArgs> args = make_shared<ChmctsManagerArgs>(thts_env);
    args->seed = 60415;
    args->max_depth = walk_len * 4;
    args->mcts_mode = false;
    args->bias = bias;
    args->num_backups_before_allowed_to_split = num_backups_before_allowed_to_split;
    args->num_threads = num_threads;
    args->num_envs = num_threads; 
    shared_ptr<ChmctsManager> manager = make_shared<ChmctsManager>(*args);

    // Run search and time, remembering to unlock the python gil if we have one, so subthreads can grab GIL to make 
    // subinterpreters
    shared_ptr<const State> init_state = thts_env->get_initial_state_itfc();
    shared_ptr<ChmctsDNode> root_node = make_shared<ChmctsDNode>(manager, init_state, 0, 0);
    // shared_ptr<ThtsPool> thts_pool = make_shared<PyThtsPool>(manager, root_node, num_threads);
    shared_ptr<ThtsPool> thts_pool = make_shared<MoThtsPool>(manager, root_node, num_threads);
    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();
    // py::gil_scoped_release rel;
    thts_pool->run_trials(num_trials);
    std::chrono::duration<double> dur = chrono::system_clock::now() - start_time;

    // Print out a tree
    // Make sure have gil again if using python objects
    // py::gil_scoped_acquire acq;
    cout << "CHMCTS with " << num_threads << " threads (took " << dur.count() << ")";
    if (print_tree_depth > 0) {
        cout << " and looks like:\n";
        cout << root_node->get_pretty_print_string(print_tree_depth);
    } 
    cout << endl << endl; 

    // Convex hull
    cout << "CH at root:" << endl << root_node->get_convex_hull_pretty_print_string() << endl << endl;

    // Pretty ball lists
    cout << "Printing convex hulls for first decision." << endl << endl;
    ThtsEnvContext ctx;
    shared_ptr<ActionVector> actions = thts_env->get_valid_actions_itfc(init_state,ctx);
    for (shared_ptr<const Action> action : *actions) {
        cout << "CH for action " << *action << ":" << endl;
        cout << root_node->get_child_node(action)->get_convex_hull_pretty_print_string() << endl << endl;
    }

    // Test out Mo MC Eval
    int num_eval_rollouts = 250;
    shared_ptr<EvalPolicy> policy = make_shared<EvalPolicy>(root_node, thts_env, manager);
    MoMCEvaluator mo_mc_eval(
        policy,  
        manager->max_depth,
        manager,
        Eigen::ArrayXd::Zero(2)-walk_len,
        Eigen::ArrayXd::Zero(2)-0.5*walk_len);
    // py::gil_scoped_release rel2;
    mo_mc_eval.run_rollouts(num_eval_rollouts, num_threads);

    cout << "CHMCTS evaluations from MoMCEval." << endl;
    cout << "Mean MO return." << endl;
    cout << mo_mc_eval.get_mean_mo_return() << endl;
    cout << "Mean MO ctx return." << endl;
    cout << mo_mc_eval.get_mean_mo_ctx_return() << endl;
    cout << "Mean MO normalised ctx return." << endl;
    cout << mo_mc_eval.get_mean_mo_normalised_ctx_return() << endl;
}

void chmcts_4d_test() { 

    // params
    double bias = 4.0;
    int num_backups_before_allowed_to_split = 10;

    int walk_len = 5;
    double stay_prob = 0.0;

    int num_trials = 10000;
    int print_tree_depth = 2;
    int num_threads = 4;

    // Setup env 
    shared_ptr<thts::test::TestMoThtsEnv> thts_env = make_shared<thts::test::TestMoThtsEnv>(walk_len, stay_prob, true);

    // Make thts manager 
    shared_ptr<ChmctsManagerArgs> args = make_shared<ChmctsManagerArgs>(thts_env);
    args->seed = 60415;
    args->max_depth = walk_len * 4;
    args->mcts_mode = false;
    args->bias = bias;
    args->num_backups_before_allowed_to_split = num_backups_before_allowed_to_split;
    args->num_threads = num_threads;
    args->num_envs = num_threads; 
    shared_ptr<ChmctsManager> manager = make_shared<ChmctsManager>(*args);

    // Run search and time, remembering to unlock the python gil if we have one, so subthreads can grab GIL to make 
    // subinterpreters
    shared_ptr<const State> init_state = thts_env->get_initial_state_itfc();
    shared_ptr<ChmctsDNode> root_node = make_shared<ChmctsDNode>(manager, init_state, 0, 0);
    // shared_ptr<ThtsPool> thts_pool = make_shared<PyThtsPool>(manager, root_node, num_threads);
    shared_ptr<ThtsPool> thts_pool = make_shared<MoThtsPool>(manager, root_node, num_threads);
    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();
    // py::gil_scoped_release rel;
    thts_pool->run_trials(num_trials);
    std::chrono::duration<double> dur = chrono::system_clock::now() - start_time;

    // Print out a tree
    // Make sure have gil again if using python objects
    // py::gil_scoped_acquire acq;
    cout << "CHMCTS with " << num_threads << " threads (took " << dur.count() << ")";
    if (print_tree_depth > 0) {
        cout << " and looks like:\n";
        cout << root_node->get_pretty_print_string(print_tree_depth);
    } 
    cout << endl << endl; 

    // Convex hull
    cout << "CH at root:" << endl << root_node->get_convex_hull_pretty_print_string() << endl << endl;

    // Pretty ball lists
    cout << "Printing convex hulls for first decision." << endl << endl;
    ThtsEnvContext ctx;
    shared_ptr<ActionVector> actions = thts_env->get_valid_actions_itfc(init_state,ctx);
    for (shared_ptr<const Action> action : *actions) {
        cout << "CH for action " << *action << ":" << endl;
        cout << root_node->get_child_node(action)->get_convex_hull_pretty_print_string() << endl << endl;
    }
    
    // Test out Mo MC Eval
    int num_eval_rollouts = 250;
    shared_ptr<EvalPolicy> policy = make_shared<EvalPolicy>(root_node, thts_env, manager);
    MoMCEvaluator mo_mc_eval(
        policy,  
        manager->max_depth,
        manager,
        Eigen::ArrayXd::Zero(4)-walk_len,
        Eigen::ArrayXd::Ones(4)/(1.0-thts_env->get_gamma()));
    // py::gil_scoped_release rel2;
    mo_mc_eval.run_rollouts(num_eval_rollouts, num_threads);

    cout << "CHMCTS evaluations from MoMCEval." << endl;
    cout << "Mean MO return." << endl;
    cout << mo_mc_eval.get_mean_mo_return() << endl;
    cout << "Mean MO ctx return." << endl;
    cout << mo_mc_eval.get_mean_mo_ctx_return() << endl;
    cout << "Mean MO normalised ctx return." << endl;
    cout << mo_mc_eval.get_mean_mo_normalised_ctx_return() << endl;
}

// void ch_lin_prog_debugging() {

//     // // Example inputs #1
//     // // Running on this outputs that the point is dominated
//     // // Got a print out of the following from 'strongly_convex_dominated' function in convex hull code:
//     // // Point considering being pruned = (8.2,-12|2)
//     // // And set of reference points = {(8.2,-12|2),(11.5,-4|1),(19.6,-13|1),(22.4,-79|1),(15.1,-8|0)}
//     // // -> this doesn't get solved somehow

//     // // Recreate points
//     // Eigen::ArrayXd p0_v(2);
//     // p0_v << 8.2, -12.0;
//     // string p0_t = "2";
//     // TaggedPoint<string> p0(p0_v,p0_t);

//     // Eigen::ArrayXd p1_v(2);
//     // p1_v << 11.5, -4.0;
//     // string p1_t = "1";
//     // TaggedPoint<string> p1(p1_v,p1_t);

//     // Eigen::ArrayXd p2_v(2);
//     // p2_v << 19.6, -13.0;
//     // string p2_t = "1";
//     // TaggedPoint<string> p2(p2_v,p2_t);

//     // Eigen::ArrayXd p3_v(2);
//     // p3_v << 22.4, -79.0;
//     // string p3_t = "1";
//     // TaggedPoint<string> p3(p3_v,p3_t);

//     // Eigen::ArrayXd p4_v(2);
//     // p4_v << 15.1, -8.0;
//     // string p4_t = "0";
//     // TaggedPoint<string> p4(p4_v,p4_t);

//     // // Input to 'strongly_convex_dominated' function
//     // TaggedPoint<string> point = p0;
//     // unordered_set<TaggedPoint<string>> ref_points = {p0,p1,p2,p3,p4};

//     // Example inputs #2
//     // Point considering being pruned = (14,-19|0)
//     // And set of reference points = {(11.5,-4|1),(20.3,-37|1),(14,-19|0),(8.2,-2|1),(16.1,-16|1),(8.2,-3|0),(15.1,-11|1)}
//     // And lp.primalType() == 1

//     // Recreate points
//     Eigen::ArrayXd p0_v(2);
//     p0_v << 14.0, -19.0;
//     string p0_t = "0";
//     TaggedPoint<string> p0(p0_v,p0_t);

//     Eigen::ArrayXd p1_v(2);
//     p1_v << 11.5, -4.0;
//     string p1_t = "1";
//     TaggedPoint<string> p1(p1_v,p1_t);

//     Eigen::ArrayXd p2_v(2);
//     p2_v << 20.3, -37.0;
//     string p2_t = "1";
//     TaggedPoint<string> p2(p2_v,p2_t);

//     Eigen::ArrayXd p3_v(2);
//     p3_v << 8.2, -2.0;
//     string p3_t = "1";
//     TaggedPoint<string> p3(p3_v,p3_t);

//     Eigen::ArrayXd p4_v(2);
//     p4_v << 16.1, -16.0;
//     string p4_t = "1";
//     TaggedPoint<string> p4(p4_v,p4_t);

//     Eigen::ArrayXd p5_v(2);
//     p5_v << 8.2, -3.0;
//     string p5_t = "0";
//     TaggedPoint<string> p5(p5_v,p5_t);

//     Eigen::ArrayXd p6_v(2);
//     p6_v << 15.1, -11.0;
//     string p6_t = "1";
//     TaggedPoint<string> p6(p6_v,p6_t);

//     // Input to 'strongly_convex_dominated' function
//     TaggedPoint<string> point = p0;
//     unordered_set<TaggedPoint<string>> ref_points = {p0,p1,p2,p3,p4,p5,p6};



//     // C&P of strongly_convex_dominated to play around with it (but print statements instead of returns)
//     // And setting T=string
//     {
//         // Base case where lp will be unbounded and would throw an error
//         if (ref_points.size() == 0 || (ref_points.size() == 1 && ref_points.contains(point))) {
//             cout << "Base case retuning false" << endl;
//             return;
//             // return false;
//         }

//         // Make lp
//         // Get n (number of points in 'ref_points' and dimension of vectors)
//         lemon::Lp lp;
//         int dim = point.point.size();

//         // Add variables for w and x
//         vector<lemon::Lp::Col> w;
//         for (int i=0; i<dim; i++) {
//             w.push_back(lp.addCol());
//             lp.colLowerBound(w[i], 0.0);
//             lp.colUpperBound(w[i], 1.0);
//         }
//         lemon::Lp::Col x = lp.addCol();

//         // Add row constrains for the inequality constraint above (take care to not include 'point')
//         for (const TaggedPoint<string>& ref_p : ref_points) {
//             if (ref_p == point) continue;
//             Eigen::ArrayXd diff = point.point - ref_p.point; // p-p_k

//             lemon::Lp::Expr row_expr = 0;
//             for (int i=0; i<dim; i++) {
//                 row_expr += diff[i] * w[i];
//             }
//             row_expr += -1.0 * x;
//             lemon::Lp::Constr row_constr = (row_expr >= 0.0);
//             lp.addRow(row_constr);
//         }

//         // Add row constraint for the equality constraint
//         lemon::Lp::Expr row_expr = 0;
//         for (int i=0; i<dim; i++) {
//             row_expr += w[i];
//         };
//         lemon::Lp::Constr row_constr = (row_expr == 1.0);
//         lp.addRow(row_constr);

//         // Set objective (max x)
//         lp.max();
//         lp.obj(x);

//         // Solve 
//         lp.solve();
//         if (lp.primalType() != lemon::Lp::OPTIMAL) {
//             cout << "Getting error in linear programming solver." << endl;
//             cout << "Point considering being pruned = " << point << endl;
//             cout << "And set of reference points = " 
//                  << thts::helper::unordered_set_pretty_print_string(ref_points) << endl;
//             cout << "And lp.primalType() == " << lp.primalType() << endl;
//             throw runtime_error("Lin prog in convex hull cant be solved. If not optimal its infeasible or unbounded");
//         }

//         // Check if optimal value was negative (meaning its dominated) or not
//         cout << "Point is strongly convex dominated?" << (lp.primal() <= 0.0) << endl;
//         // return lp.primal() <= 0.0;
//     }
// }

// string eigen_6d_to_string(Eigen::ArrayXd& v) {
//     stringstream ss;
//     ss << "[" << v[0] << "," << v[1] << "," << v[2] << "," << v[3] << "," << v[4] << "," << v[5] << "]";
//     return ss.str();
// }

// void compare_czt_bts_fruit_tree() {
//     py::gil_scoped_release release;

//     // params
//     int search_time = 30;
//     int num_threads = 8;
//     string thts_unique_filename = "/bin";
//     string thts_unique_filename_bts = "/usr";

//     // Setup env 
//     string mo_gym_env_id = "fruit-tree-v0";
//     shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
//     shared_ptr<MoGymMultiprocessingThtsEnv> thts_env = make_shared<MoGymMultiprocessingThtsEnv>(
//         pickle_wrapper, thts_unique_filename, mo_gym_env_id);
//     shared_ptr<MoGymMultiprocessingThtsEnv> thts_env_bts = make_shared<MoGymMultiprocessingThtsEnv>(
//         pickle_wrapper, thts_unique_filename_bts, mo_gym_env_id);

//     // Make thts manager 
//     shared_ptr<CztManagerArgs> args = make_shared<CztManagerArgs>(thts_env);
//     args->seed = 60415;
//     args->mcts_mode = false;
//     args->bias = 30.0;
//     args->num_backups_before_allowed_to_split = 50;
//     args->num_threads = num_threads;
//     args->num_envs = num_threads;
//     shared_ptr<CztManager> manager = make_shared<CztManager>(*args);

//     Eigen::ArrayXd default_val = Eigen::ArrayXd::Zero(6);
//     shared_ptr<SmtBtsManagerArgs> sargs = make_shared<SmtBtsManagerArgs>(thts_env_bts, default_val);
//     sargs->seed = 60415;
//     sargs->mcts_mode = false;
//     sargs->simplex_node_l_inf_thresh = 0.05;
//     sargs->simplex_node_split_visit_thresh = 1;
//     sargs->temp = 100.0;
//     sargs->epsilon = 0.5;
//     sargs->num_threads = num_threads;
//     sargs->num_envs = num_threads; 
//     shared_ptr<SmtBtsManager> smanager = make_shared<SmtBtsManager>(*sargs);

//     // Setup python servers
//     for (int i=0; i<args->num_envs; i++) {
//         PyMultiprocessingThtsEnv& py_mp_env = *dynamic_pointer_cast<PyMultiprocessingThtsEnv>(manager->thts_env(i));
//         py_mp_env.start_python_server(i);
//     }
//     for (int i=0; i<args->num_envs; i++) {
//         PyMultiprocessingThtsEnv& py_mp_env = *dynamic_pointer_cast<PyMultiprocessingThtsEnv>(smanager->thts_env(i));
//         py_mp_env.start_python_server(i);
//     }

//     // Run searches
//     shared_ptr<const State> init_state = thts_env->get_initial_state_itfc();

//     cout << "starting czt search" << endl; 
//     shared_ptr<CztDNode> root_node = make_shared<CztDNode>(manager, init_state, 0, 0);
//     shared_ptr<ThtsPool> thts_pool = make_shared<MoThtsPool>(manager, root_node, num_threads);
//     thts_pool->run_trials(numeric_limits<int>::max(), search_time);
//     cout << "czt search finished" << endl; 

//     cout << "starting bts search" << endl; 
//     shared_ptr<SmtBtsDNode> sroot_node = make_shared<SmtBtsDNode>(smanager, init_state, 0, 0);
//     shared_ptr<ThtsPool> sthts_pool = make_shared<MoThtsPool>(smanager, sroot_node, num_threads);
//     sthts_pool->run_trials(numeric_limits<int>::max(), search_time);
//     cout << "bts search finished" << endl; 

//     // Simulate trials until CZT and BTS recommend different actions and then print out some debugging info
//     int num_sim = 1000;
//     int max_print_out = 5;
//     int print_outs = 0;

//     cout << "starting simulations to find discrepancies" << endl; 
//     for (int i=0; i<num_sim; i++) {
//         if (print_outs >= max_print_out) {
//             break;
//         }

//         shared_ptr<MoThtsContext> mo_context = static_pointer_cast<MoThtsContext>(
//             thts_env->sample_context_itfc(0, *manager));
//         manager->register_thts_context(0, mo_context);
//         smanager->register_thts_context(0, mo_context);

//         thts_env->reset_itfc();

//         shared_ptr<const State> cur_state = init_state;
//         shared_ptr<CztDNode> cur_czt_node = root_node;
//         shared_ptr<SmtBtsDNode> cur_bts_node = sroot_node;

//         int depth = 0;

//         // Run trial
//         while (!thts_env->is_sink_state_itfc(cur_state, *mo_context)) {
//             shared_ptr<const Action> czt_action = cur_czt_node->recommend_action(*mo_context);
//             shared_ptr<const Action> bts_action = cur_bts_node->recommend_action(*mo_context);

//             if (czt_action != bts_action) {
//                 cout << "----------" << endl;
//                 cout << "found instance of czt_action != bts_action, print debug info out here. (Moving onto next trial for now, this was trial number " << i << ")" << endl;
//                 print_outs++;
//                 cout << "Current depth is: " << depth << endl;
//                 Eigen::ArrayXd& ctx = mo_context->context_weight;
//                 cout << "Context weight is: " << eigen_6d_to_string(ctx) << endl;

//                 shared_ptr<NGV> ngv = cur_bts_node->simplex_map.get_leaf_tn_node(ctx)->get_closest_ngv_vertex(ctx);
//                 cout << "Closes NGV weight in BTS dnode is: " << eigen_6d_to_string(ngv->weight) << endl;
//                 cout << "Closes NGV value_estimate in BTS dnode is: " << eigen_6d_to_string(ngv->value_estimate) << endl;
//                 cout << "(Contextual value = " << thts::helper::dot(ngv->value_estimate,ctx) << ")" << endl;
//                 cout << "(Dist from weight to selected ngv = " << thts::helper::dist(ngv->weight,ctx) << ")" << endl;

//                 double min_dist = numeric_limits<double>::max();
//                 Eigen::ArrayXd min_dist_w = Eigen::ArrayXd::Zero(6);
//                 for (shared_ptr<NGV> node : *cur_bts_node->simplex_map.n_graph_vertices) {
//                     double dist = thts::helper::dist(node->weight,ctx);
//                     if (dist < min_dist) {
//                         min_dist = dist;
//                         min_dist_w = node->weight;
//                     }
//                 }
//                 cout << "(Min dist to any ngv is: " << min_dist << " with weight " << eigen_6d_to_string(min_dist_w) << ")" << endl;

//                 int child_num = 0;
//                 shared_ptr<ActionVector> actions = thts_env->get_valid_actions_itfc(cur_state, *mo_context);
//                 for (shared_ptr<const Action> action : *actions) {
//                     if (!cur_bts_node->has_child_node_itfc(action)) {
//                         continue;
//                     }
//                     shared_ptr<SmtBtsCNode> child_node = cur_bts_node->get_child_node(action);
//                     shared_ptr<NGV> child_ngv = child_node->simplex_map.get_leaf_tn_node(ctx)->get_closest_ngv_vertex(ctx);
//                     cout << "Closest NGV weight in " << child_num << "th BTS child is: " << eigen_6d_to_string(child_ngv->weight) << endl;
//                     cout << "Closes NGV value_estimate in " << child_num << "th BTS child is: " << eigen_6d_to_string(child_ngv->value_estimate) << endl;
//                     cout << "(Contextual value = " << thts::helper::dot(child_ngv->value_estimate,ctx) << ")" << endl;
//                     cout << "(Dist from weight to selected ngv = " << thts::helper::dist(child_ngv->weight,ctx) << ")" << endl;

//                     double chld_min_dist = numeric_limits<double>::max();
//                     Eigen::ArrayXd chld_min_dist_w = Eigen::ArrayXd::Zero(6);
//                     for (shared_ptr<NGV> node : *child_node->simplex_map.n_graph_vertices) {
//                         double dist = thts::helper::dist(node->weight,ctx);
//                         if (dist < chld_min_dist) {
//                             chld_min_dist = dist;
//                             chld_min_dist_w = node->weight;
//                         }
//                     }
//                     cout << "(Min dist to any ngv is: " << chld_min_dist << " with weight " << eigen_6d_to_string(chld_min_dist_w) << ")" << endl;
                    
//                     child_num++;
//                 }

//                 cout << "Todo print out CZT stuff" << endl;
//                 cout << "----------" << endl << endl;
//                 break;
//             }

//             shared_ptr<CztCNode> c_czt_node = cur_czt_node->get_child_node(czt_action);
//             shared_ptr<SmtBtsCNode> c_bts_node = cur_bts_node->get_child_node(bts_action);

//             cur_state = thts_env->sample_transition_distribution_itfc(
//                 cur_state, bts_action, *manager, *mo_context);
//             shared_ptr<const Observation> cur_obs = static_pointer_cast<const Observation>(cur_state);

//             if (!c_czt_node->has_child_node_itfc(cur_obs) || !c_bts_node->has_child_node_itfc(cur_obs)) {
//                 // Got outside of the tree
//                 cout << "Breaking out of trial " << i << "because exited one of the trees" << endl;
//                 break;
//             }

//             cur_czt_node = c_czt_node->get_child_node(cur_state);
//             cur_bts_node = c_bts_node->get_child_node(cur_state);

//             depth++;
//         }
//     }
// }

// C++ entry point for debugging
int main(int argc, char *argv[]) {
    py::scoped_interpreter py_interpreter;

    /**
     * Debugging tests
    */
    // pickle_test();
    // sem_test();
    // pickle_multiproc_test();
    // pickle_subinterpret_test(); // commented out now
    // shared_mem_test();
    // shared_mem_wrapper_test();
    // shared_mem_destroy_test();
    // pybind_gil_test();

    /**
     * Testing py thts env
    */
    // bool bts_alpha = 1.0;
    // bool use_python_env = true;
    // py_thts_env_test(bts_alpha, use_python_env); 

    /**
     * Testing czt
    */
    // czt_test();
    // czt_4d_test();

    /**
     * Testing python gym envs 
     * TODO: this currently fails, because gym envs requires algorithms to run in a model free mode, but we only have 
     *      single objective algorithms implemented in a planning mode
    */
    // gym_env_test();

    /**
     * Testing python mo gym envs
    */
    // mo_gym_env_test();

    /**
     * Testing Eigen SVD
    */
    // eigen_svd_test();

    /**
     * Test simplex map
    */
    // sm_bts_test();
    // sm_bts_4d_test();
    sm_bts_bin_tree_test();
    sm_bts_bin_tree_4d_test();
    sm_dents_test();

    /**
     * Testing chmcts
    */
    chmcts_test();
    chmcts_4d_test();

    /**
     * Debugging Convex hull linear programs
     */
    // ch_lin_prog_debugging();

    /**
     * Testing being able to run multiple thts instances at once
     * 
     * NB: argv[1] needs to be an existing filename. 
     * I tried running with ./pyex /usr and ./pyex /bin simultaneously
     * And tested trying to run a second ./pyex /usr gets a runtime error
    */
    // mo_gym_env_test(argv[1]);

    /**
     * Debugging simplex maps on fruit tree
     */
    // compare_czt_bts_fruit_tree();

    return 0;
}

PYBIND11_MODULE(thts, m) { 

    // Module docstring
    m.doc() = "python module to access the THTS++ library";
};





