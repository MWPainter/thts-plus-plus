#include "module.h"

#include "test_env.h"

#include "py/gym_multiprocessing_thts_env.h"
#include "py/mo_gym_multiprocessing_thts_env.h"
#include "py/mo_py_mc_eval.h"
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

void py_thts_env_test(double alpha, bool use_python_env) {
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
        manager = make_shared<DentsManager>(args);

        // Setup python servers
        if (use_python_env) {
            for (int i=0; i<args.num_envs; i++) {
                PyMultiprocessingThtsEnv& py_mp_env = *dynamic_pointer_cast<PyMultiprocessingThtsEnv>(manager->thts_env(i));
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

    // Keep on crashing after here
    // TODO: sort out for python lib release, because need to not crash at least until Py_Finalize to use in Python code
    // Annoying to run ipcrm -v -a all the time, so make sure thats not necessary by the time it crashes

    if (use_python_env) {
        for (int i=0; i<num_threads; i++) {
            PyMultiprocessingThtsEnv& py_mp_env = *dynamic_pointer_cast<PyMultiprocessingThtsEnv>(manager->thts_env(i));
            py_mp_env.clear_unix_sem_and_shm();
        }
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
    // params
    string gym_env_id = "FrozenLake-v1";
    int num_trials = 100;
    int print_tree_depth = 2;
    int num_threads = 4;
    double alpha = 1.0;

    // Make py env (making a py::object of python thts env, and pass into constructor)
    shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
    shared_ptr<ThtsEnv> thts_env = make_shared<GymMultiprocessingThtsEnv>(pickle_wrapper, gym_env_id);

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
    shared_ptr<ThtsPool> thts_pool = make_shared<PyThtsPool>(manager, root_node, num_threads);

    // Run thts trials (same as c++)
    // Needs to not have the gil, so threads can grab it any make interpreters
    py::gil_scoped_release rel;
    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();
    thts_pool->run_trials(num_trials);
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

    // Keep on crashing after here
    // TODO: sort out for python lib release, because need to not crash at least until Py_Finalize to use in Python code
    // Annoying to run ipcrm -v -a all the time, so make sure thats not necessary by the time it crashes
    for (int i=0; i<num_threads; i++) {
        PyMultiprocessingThtsEnv& py_mp_env = *dynamic_pointer_cast<PyMultiprocessingThtsEnv>(manager->thts_env(i));
        py_mp_env.clear_unix_sem_and_shm();
    }

    // Force tree destructors to be called (to clean up python objects with the gil)
    // resetting smart pointers should refcount to zero and call destructor
    // A bit annoying having to make everything a smart pointer to call destructors with gil, but mech
    manager.reset();
    thts_env.reset();
    thts_pool.reset();
    root_node.reset();
}

void mo_gym_env_test() {
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
        pickle_wrapper, mo_gym_env_id);

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
    shared_ptr<ThtsPool> thts_pool = make_shared<MoPyThtsPool>(manager, root_node, num_threads);
    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();
    py::gil_scoped_release rel;
    thts_pool->run_trials(num_trials);
    thts_pool->run_trials(num_trials);
    std::chrono::duration<double> dur = chrono::system_clock::now() - start_time;

    // Print out a tree
    // Make sure have gil again if using python objects
    py::gil_scoped_acquire acq;
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
    MoPyMCEvaluator mo_mc_eval(
        policy, 
        manager->max_depth,
        manager,
        r_min,
        r_max);
    py::gil_scoped_release rel2;
    mo_mc_eval.run_rollouts(num_eval_rollouts, num_threads);
    py::gil_scoped_acquire acq2;

    cout << "CZT evaluations from MoMCEval." << endl;
    cout << "Mean MO return." << endl;
    cout << mo_mc_eval.get_mean_mo_return() << endl;
    cout << "Mean MO ctx return." << endl;
    cout << mo_mc_eval.get_mean_mo_ctx_return() << endl;
    cout << "Mean MO normalised ctx return." << endl;
    cout << mo_mc_eval.get_mean_mo_normalised_ctx_return() << endl;

    // Trying to make python embedding exit gracefully stuff (see py_thts_env_test to understand)
    for (int i=0; i<num_threads; i++) {
        PyMultiprocessingThtsEnv& py_mp_env = *dynamic_pointer_cast<PyMultiprocessingThtsEnv>(manager->thts_env(i));
        py_mp_env.clear_unix_sem_and_shm();
    }
    manager.reset();
    args.reset();
    thts_env.reset();
    thts_pool.reset();
    root_node.reset();
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
    args->use_triangulation = true;
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
    args->use_triangulation = true;
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
    args->simplex_node_max_depth = 3;
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

// C++ entry point for debugging
int main(int argc, char *argv[]) {
    py::scoped_interpreter guard;

    /**
     * Debugging tests
    */
    // pickle_test();
    // sem_test();
    // pickle_multiproc_test();
    // pickle_subinterpret_test();
    // shared_mem_test();
    // shared_mem_wrapper_test();

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
    // sm_dents_test();
    // sm_bts_bin_tree_test();
    // sm_bts_bin_tree_4d_test();

    /**
     * Testing chmcts
    */
    chmcts_test();
    chmcts_4d_test();

    return 0;
}

PYBIND11_MODULE(thts, m) { 

    // Module docstring
    m.doc() = "python module to access the THTS++ library";
};