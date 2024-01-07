#pragma once

#include "thts.h"

#include "thts_decision_node.h"
#include "thts_env_context.h"
#include "thts_logger.h"
#include "thts_manager.h"

#include <chrono>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>


namespace thts::python {
    using namespace thts;

    /**
     * A class encapsulating all of the logic required to run a thts routine.
     * 
     * At a higher level, this class creats a pool of threads that call 'run_trial' until they are signalled to stop. 
     * The threads are signaled to stop when the maximum duration has passed, or the desirect number of trials have 
     * been performed.
     * 
     * All functions are marked as virtual so that they can be mocked for unit testing.
     * 
     * Member variables:
     */
    class PyThtsPool : public ThtsPool {
        protected:   


        public:
            /**
             * Constructs the ThtsPool with 'num_threads' worker threads.
             * 
             * Args:
             *      manager: The ThtsManager to use for this instance of thts
             *      root_node: 
             *          The root node to run thts on. The algorithm is specified by the subclass of ThtsDNode used. If 
             *          NULL, then a default root node construction is attempted using the initial state from 
             *          thts_manager->thts_env.
             *      num_threads: The number of worker threads to spawn
             */
            PyThtsPool(
                std::shared_ptr<ThtsManager> thts_manager=nullptr, 
                std::shared_ptr<ThtsDNode> root_node=nullptr, 
                int num_threads=1,
                std::shared_ptr<ThtsLogger> logger=nullptr);

            /**
             * Destructor. Required to allow the thread pool to exit gracefully.
             */
            virtual ~PyThtsPool() = default;

        protected:
            /**
             * The worker thread thnuk.
             * 
             * Waits for work, and calls 'run_thts_trial' until there is no more work to do.
             */
            virtual void worker_fn() override;           
    };
}
