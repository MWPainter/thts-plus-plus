#include "py/py_thts.h"

#include "py/py_helper.h"

#include <Python.h>
#include <pybind11/pybind11.h>

using namespace std;
using namespace thts;
namespace py = pybind11;

namespace thts::python {
    /**
     * Constructor.
     * 
     * The following steps will occur:
     * - initialises member variables
     * - creates a root node if needed
     * - spawns worker threads
     * - worker threads will wait on can_run_trial_cv on first loop
     *      (given the initialisations (trials_remaining==0), the call can_run_trial() will return false)
     * - current thread waits on 'can_run_trial_cv', to wait until threads are all waiting on the cv
     *      (subtle note: becausse workers hold work_left_lock when they call notify_all, the thread running this 
     *          constructor will not be able to grab the lock until it waits on the work_left_cv)
     */
    PyThtsPool::PyThtsPool(
        shared_ptr<ThtsManager> thts_manager, 
        shared_ptr<ThtsDNode> root_node, 
        int num_threads, 
        shared_ptr<ThtsLogger> logger,
        bool start_threads_in_this_constructor) :
            ThtsPool(thts_manager, root_node, num_threads, logger, false) 
    {
        if (start_threads_in_this_constructor) {
            for (int i=0; i<num_threads; i++) {
                workers[i] = thread(&PyThtsPool::worker_fn, this, i);
            }
        }
    }


    /**
     * The worker thread function.
     * 
     * Copied from thts.cpp
     * 
     * Added setting up python interpreter in 'setup thread'
     */
    void PyThtsPool::worker_fn(int tid) {
        // setup thread
        thts_manager->register_thread_id(tid);
        // Make new interpreter
        // Need to lock with CPython API because pybind11 gil interface not built to work with it
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

        // main work loop
        lock_guard<mutex> lg(work_left_lock);
        while (thread_pool_alive) {
            num_threads_working--;

            if (!work_left()) {
                work_left_cv.notify_all();
            }
            while (!work_left()) {
                work_left_cv.wait(work_left_lock);
                if (!thread_pool_alive) return;
            }

            num_threads_working++;
            trials_remaining--;
            int trials_remaining_copy = trials_remaining;

            work_left_lock.unlock();
            run_thts_trial(trials_remaining_copy, tid);
            work_left_lock.lock();
        }
    }
}