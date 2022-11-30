#pragma once

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


namespace thts {
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
     *      workers: A vector of threads (thread pool) that run the worker_fn routine.
     *      work_left_cv: A condition variable used to coordinate when workers are working.
     *      work_left_lock: A mutex for 'work_left_lock', protecting variables used to decide when there is work.
     *      logging_lock: A mutex protecting any logging performed by thts.
     *      thread_pool_alive: A boolean stating if the workers thread pool is running. Set to false at destruction.
     *      num_threads: The number of threads used in the workers thread pool.
     *      num_trials: The number of trials the pool is currently trying to run in total.
     *      start_time: The start time of a 'run_trials' call.
     *      max_run_time: The maximum duration we allow thts to run for)not including finishing off the current trials).
     *      trials_remaining: The number of trials the workers pool needs to run to have completed 'num_trials' trials.
     *      num_threads_working: The number of threads currently working
     *      trials_completed: The number of trials completed for 
     *      thts_manager: The ThtsManager to use in the thts planning routine
     *      root_node: The ThtsDNode root node that currently want to plan for
     */
    class ThtsPool {
        protected:   
            // multithreading variables (also protected by can_work_lock)
            std::vector<std::thread> workers;
            std::condition_variable_any work_left_cv;
            std::mutex work_left_lock;
            std::mutex logging_lock;
            bool thread_pool_alive;

            // constant after init
            int num_threads;

            // protected by can_work_lock - variables only updated on 'run_trials' call 
            int num_trials;
            std::chrono::time_point<std::chrono::system_clock> start_time;
            std::chrono::duration<double> max_run_time;

            // protected by can_work_lock - variables related to if should run more trials + updated by workers
            int trials_remaining;
            int num_threads_working;

            // protected by logging_lock - variables to do with logging
            int trials_completed;
            std::shared_ptr<ThtsLogger> logger;

            // Manager and root node specifying the flavour of thts to run (the problem and algorithm)
            std::shared_ptr<ThtsManager> thts_manager;
            std::shared_ptr<ThtsDNode> root_node;

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
            ThtsPool(
                std::shared_ptr<ThtsManager> thts_manager=nullptr, 
                std::shared_ptr<ThtsDNode> root_node=nullptr, 
                int num_threads=1,
                std::shared_ptr<ThtsLogger> logger=nullptr);

            /**
             * Destructor. Required to allow the thread pool to exit gracefully.
             */
            virtual ~ThtsPool();

            /**
             * Returns a boolean for if the workers need to do more work to complete a 'run_trials' call.
             * 
             * Calls to this function should be protected using work_left_lock.
             */
            virtual bool work_left();

        protected:
            /**
             * Checks if a worker should continue their selection phase or if it is time to end.
             * 
             * Selection phase ends when a leaf node (in the thts_env) or max depth is hit, or when running in mcts_mode, 
             * once any new nodes have been added.
             * 
             * Args:
             *      cur_node: The most recent node reached in the selection phase
             *      new_decision_node_created_this_trial: If a new decision node has been created this trial
             * 
             * Returns:
             *      If the selection phase should be ended.
             */
            virtual bool should_continue_selection_phase(
                std::shared_ptr<ThtsDNode> cur_node, bool new_decision_node_created_this_trial);

            /**
             * Runs the selection phase of a trial, called by worker threads.
             * 
             * Args:
             *      nodes_to_backup: 
             *          A list to be filled with pairs of (ThtsDNode, ThtsCNode), that should have 'backup' called on them in the 
             *          backup phase
             *      rewards:
             *          A list to be filled with rewards obtained during this selection phase (including the heuristic 
             *          value of a frontier node (which is not visited))
             *      context:
             *          The ThtsContext for this trial
             * 
             * Returns:
             *      Nothing, 'nodes_to_backup' and 'rewards' are filled by this function as 'return_values'.
             */
            void run_selection_phase(
                std::vector<std::pair<std::shared_ptr<ThtsDNode>,std::shared_ptr<ThtsCNode>>>& nodes_to_backup, 
                std::vector<double>& rewards, 
                ThtsEnvContext& context);

            /**
             * Runs the backup phase of a trial, called by worker threads.
             * 
             * Args:
             *      nodes_to_backup: 
             *          A list of pairs of (ThtsDNode, ThtsCNode), to call backup on (created by selection phase)
             *      rewards:
             *          A list of rewards obtained during the selection phase that may be used by backups
             *      context:
             *          The ThtsContext for this trial
             */
            void run_backup_phase(
                std::vector<std::pair<std::shared_ptr<ThtsDNode>,std::shared_ptr<ThtsCNode>>>& nodes_to_backup, 
                std::vector<double>& rewards, 
                ThtsEnvContext& context);

            /**
             * Performs a single thts trial. Called by worker_fn.
             * 
             * Args:
             *      trials_remaining: 
             *          The number of trials remaining at the time of calling (not including the one about to be run by 
             *          this function)
             */
            virtual void run_thts_trial(int trials_remaining);

            /**
             * The worker thread thnuk.
             * 
             * Waits for work, and calls 'run_thts_trial' until there is no more work to do.
             */
            virtual void worker_fn();

        public:
            /**
             * Waits on the ThtsPool to finish a 'run_trials' call. Returns when the workers have finished.
             */
            virtual void join();

            /**
             * Specifys how many trails/how long to run trials of thts for.
             * 
             * Default values essentially runs the trials indefinitely.
             * 
             * Args:
             *      max_trials: The maximum number of thts trials to run
             *      max_time: The maximum (human time) to run thts trials for
             *      blocking: If this call is blocking and will only return when the thts trials have finished
             */
            virtual void run_trials(
                int max_trials=std::numeric_limits<int>::max(), 
                double max_time=std::numeric_limits<double>::max(), 
                bool blocking=true);            
    };
}

