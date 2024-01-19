#include "py/py_mc_eval.h"

#include "py/py_helper.h"
#include <Python.h>

using namespace std;
using namespace thts;


/**
 * MC Eval implementation
*/
namespace thts::python {
    PyMCEvaluator::PyMCEvaluator(
        shared_ptr<EvalPolicy> policy, 
        int max_trial_length, 
        shared_ptr<ThtsManager> manager) :
            MCEvaluator(policy,max_trial_length,manager)
    {
    }
    
    /**
    */
    void PyMCEvaluator::thread_run_rollouts(
        int total_rollouts, int thread_id, int num_threads, shared_ptr<EvalPolicy> thread_policy) 
    {

        // Setup Py subinterpreter
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

        // Run normal run rollouts
        return MCEvaluator::thread_run_rollouts(total_rollouts, thread_id, num_threads, thread_policy);
    }

    /**
     * Make run rollouts call our python version of thread_run_rollouts
    */
    void PyMCEvaluator::run_rollouts(int num_rollouts, int num_threads) {
        // setup vars
        num_rollouts_to_run = num_rollouts;
        num_rollouts_started = 0;

        // spawn
        vector<thread> threads;
        for (int i=0; i<num_threads; i++) {
            shared_ptr<EvalPolicy> thread_eval_policy = make_shared<EvalPolicy>(*policy,manager->thts_env(i));
            threads.push_back(thread(
                &PyMCEvaluator::thread_run_rollouts, 
                this, 
                num_rollouts, 
                i, 
                num_threads, 
                thread_eval_policy));
        }

        // wait
        for (int i=0; i<num_threads; i++) {
            threads[i].join();
        }

    }
}