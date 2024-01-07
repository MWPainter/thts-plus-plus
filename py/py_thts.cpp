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
        shared_ptr<ThtsLogger> logger) :
            ThtsPool(thts_manager, root_node, num_threads, logger, false) 
    {
        for (int i=0; i<num_threads; i++) {
            workers[i] = thread(&PyThtsPool::worker_fn, this);
        }
    }

    /**
     * Initialised a python subinterpreter for this thread, and then run the normal worker_fn
     * Note that new interpreter will release the global gil, and then acquire its local gil in Py_NewInterpreterFromConfig
     */
    void PyThtsPool::worker_fn() {
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

        // Do work
        ThtsPool::worker_fn();
    }
}