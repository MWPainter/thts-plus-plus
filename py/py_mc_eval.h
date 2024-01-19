#pragma once

#include "mc_eval.h"

#include <pybind11/pybind11.h>

namespace thts::python {
    using namespace thts;
    namespace py = pybind11;

    /**
     * Py Version of MCEvaluator
    */
    class PyMCEvaluator : virtual public MCEvaluator {
        protected:

            /**
             * Runs rollouts as a worker thread
            */
            virtual void thread_run_rollouts(
                int total_rollouts, 
                int thread_id, 
                int num_threads, 
                std::shared_ptr<EvalPolicy> thread_policy) override;



        public:
            PyMCEvaluator(
                std::shared_ptr<EvalPolicy> eval_policy,
                int max_trial_length,
                std::shared_ptr<ThtsManager> manager);
            
            virtual ~PyMCEvaluator() = default;

            /**
             * Run 'num_rollout' many rollouts to gather stats. Does so by spawning 'num_threads' many threads and 
             * setting them off to run rollouts using the 'thread_run_rollouts' function.
             * 
             * Assumes that the tree is static during this call, so does not lock the nodes of the tree.
            */
            virtual void run_rollouts(int num_rollouts, int num_threads) override;
    };
}