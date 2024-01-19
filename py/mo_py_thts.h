#pragma once

#include "py/py_thts.h"
#include "mo/mo_thts.h"


namespace thts::python {
    using namespace thts;

    /**
     * Start python worker fn threads in PyThtsPool constructor
     * run_thts_trial calls MO version of run_thts_trial
     */
    class MoPyThtsPool : public PyThtsPool, public MoThtsPool {

        public:
            /**
             * 
             */
            MoPyThtsPool(
                std::shared_ptr<ThtsManager> thts_manager=nullptr, 
                std::shared_ptr<MoThtsDNode> root_node=nullptr, 
                int num_threads=1,
                std::shared_ptr<ThtsLogger> logger=nullptr,
                bool start_threads_in_this_constructor=true);

            /**
             * Destructor. Required to allow the thread pool to exit gracefully.
             */
            virtual ~MoPyThtsPool() = default;

        protected:
            /**
             * Run trial for MO env
            */
            virtual void run_thts_trial(int trials_remaining, int tid) override;                  
    };
}
