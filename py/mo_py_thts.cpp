// #include "py/mo_py_thts.h"

// using namespace std;
// using namespace thts;

// namespace thts::python {
//     /**
//      * Constructor.
//      * 
//      * The following steps will occur:
//      * - initialises member variables
//      * - creates a root node if needed
//      * - spawns worker threads
//      * - worker threads will wait on can_run_trial_cv on first loop
//      *      (given the initialisations (trials_remaining==0), the call can_run_trial() will return false)
//      * - current thread waits on 'can_run_trial_cv', to wait until threads are all waiting on the cv
//      *      (subtle note: becausse workers hold work_left_lock when they call notify_all, the thread running this 
//      *          constructor will not be able to grab the lock until it waits on the work_left_cv)
//      */
//     MoPyThtsPool::MoPyThtsPool(
//         shared_ptr<ThtsManager> thts_manager, 
//         shared_ptr<MoThtsDNode> root_node, 
//         int num_threads, 
//         shared_ptr<ThtsLogger> logger,
//         bool start_threads_in_this_constructor) :
//             ThtsPool(thts_manager, root_node, num_threads, logger, false),
//             PyThtsPool(thts_manager, root_node, num_threads, logger, true),
//             MoThtsPool(thts_manager, root_node, num_threads, logger, false)
//     {
//     }


//     void MoPyThtsPool::run_thts_trial(int trials_remaining, int tid) {
//         MoThtsPool::run_thts_trial(trials_remaining, tid);
//     }
// }