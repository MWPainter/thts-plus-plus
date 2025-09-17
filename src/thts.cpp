#include "thts.h"

#include "thts_chance_node.h"
#include "thts_types.h"

#include <algorithm>
#include <utility>
#include <iostream>

using namespace std;

namespace thts {
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
    /**
     * CConstructor for ThtsPool for subclasses to use in their initialisation
     * For example, if override worker_fn to add setup to a thread (e.g. PyThtsPool), then would want to 
     * make threads that call PyThtsPool::worker_fn rather than ThtsPool::worker_fn
     *
     * Aditionally, remember to make sure that root_node is in the transposition table if using graph search
     */
    ThtsPool::ThtsPool(
        shared_ptr<ThtsManager> thts_manager, 
        shared_ptr<ThtsDNode> root_node, 
        int num_threads, 
        shared_ptr<ThtsLogger> logger,
        bool start_threads_in_this_constructor) :
            workers(num_threads),
            work_left_cv(),
            work_left_lock(),
            logging_lock(),
            thread_pool_alive(true),
            num_threads(num_threads),
            num_trials(0),
            start_time(std::chrono::system_clock::now()),
            max_run_time(0.0),
            trials_remaining(0),
            num_threads_working(num_threads),
            trials_completed(0),
            logger(logger),
            thts_manager(thts_manager),
            root_node(root_node)
    {
        if (thts_manager == nullptr || root_node == nullptr) {
            throw runtime_error("Cannot make ThtsPool without a thts manager, or root node");
        }
        if (thts_manager->graph_search) {
            unique_lock<shared_mutex> writer_lock(thts_manager->dmap_lock);
            thts_manager->dmap[root_node->state] = root_node;
        }
        if (start_threads_in_this_constructor) {
            for (int i=0; i<num_threads; i++) {
                workers[i] = thread(&ThtsPool::worker_fn, this, i);
            }
        }
    }

    /**
     * Destructor
     * 
     * - Protected by work_left_lock does following:
     *      - Sets thread_pool_alive to false so that worker threads will exit
     *      - Signals all worker threads so they can exit
     *          (N.B. Workers only do not hold this lock when running a trial or waiting on the cv)
     * - Waits for worker threads to exit using join
     */
    ThtsPool::~ThtsPool() {
        work_left_lock.lock();
        thread_pool_alive = false;
        work_left_cv.notify_all();
        work_left_lock.unlock();
        for (int i=0; i<num_threads; i++) {
            workers[i].join();
        }
    }

    /**
     * Locks a list of ThtsNode objects in a strict ordering using their pointers
     */
    void ThtsPool::lock_node_list(vector<shared_ptr<ThtsNode>>& nodes_to_lock) 
    {
        std::sort(nodes_to_lock.begin(), nodes_to_lock.end());
        for (shared_ptr<ThtsNode>& node : nodes_to_lock) {
            node->lock.lock();
        }
    }

    /**
     * Safely locks a dnode and its children. Accounts for being in a graph search. Nodes need to be locked in a  
     * strict ordering to avoid deadlock.
     *  
     * First get the list of nodes to lock (the node and all of its children), node->lock needs to be held for this 
     * Second locks all nodes, sorted by their pointers 
     * Third checks that the number of things we locked == number of children + 1 
     * Fourth if the check fails, then a child was added under our feet (between grabbing the list of children and 
     *      trying to lock them all). So we should unlock all the nodes we successfully locked and retry.
     */
    void ThtsPool::lock_dnode_and_children(shared_ptr<ThtsDNode> node) {
        // while (true) 
        // {
        //     vector<shared_ptr<ThtsNode>> to_lock;
        //     to_lock.push_back(node);
        //     node->lock.lock();
        //     for (auto [_action, child] : node->children)
        //     {
        //         to_lock.push_back(child);
        //     }
        //     node->lock.unlock();

        //     ThtsPool::lock_node_list(to_lock);

        //     if (node->children.size() + 1 == to_lock.size())
        //     {
        //         break;
        //     }
            
        //     for (shared_ptr<ThtsNode>& to_unlock : to_lock) 
        //     {
        //         to_unlock->lock.unlock();
        //     }
        // }

        while (true) {
            // Collect children under node->lock
            vector<shared_ptr<ThtsNode>> to_lock;
            {
                lock_guard<mutex> guard(node->lock);   // safely hold while reading children
                to_lock.push_back(node);
                for (auto& [action, child] : node->children) {
                    to_lock.push_back(child);
                }
            }

            // Now lock everything in sorted order
            lock_node_list(to_lock);

            // Verify no new children slipped in
            size_t expected = 1 + node->children.size();
            if (expected == to_lock.size()) {
                return;  // success
            }

            // Unlock and retry
            for (auto& n : to_lock) {
                n->lock.unlock();
            }
        }
    }

    /**
     * Safely unlocks a dnode and its children. Accounts for being in a graph search. Nodes need to be locked in a  
     * strict ordering to avoid deadlock. In 2 phase locking, don't need to be so careful about how unlock
     */
    void ThtsPool::unlock_dnode_and_children(shared_ptr<ThtsDNode> node) {
        node->lock.unlock();
        for (auto [_action, child] : node->children)
        {
            child->lock.unlock();
        }
    }

    /**
     * v1TODO: make this and lock_dnode one function
     */
    void ThtsPool::lock_cnode_and_children(shared_ptr<ThtsCNode> node) {
        // while (true) 
        // {
        //     vector<shared_ptr<ThtsNode>> to_lock;
        //     to_lock.push_back(node);
        //     node->lock.lock();
        //     for (auto [_obs, child] : node->children)
        //     {
        //         to_lock.push_back(child);
        //     }
        //     node->lock.unlock();

        //     ThtsPool::lock_node_list(to_lock);

        //     if (node->children.size() + 1 == to_lock.size())
        //     {
        //         break;
        //     }
            
        //     for (shared_ptr<ThtsNode>& to_unlock : to_lock) 
        //     {
        //         to_unlock->lock.unlock();
        //     }
        // }



        while (true) {
            // Collect children under node->lock
            vector<shared_ptr<ThtsNode>> to_lock;
            {
                lock_guard<mutex> guard(node->lock);   // safely hold while reading children
                to_lock.push_back(node);
                for (auto& [obs, child] : node->children) {
                    to_lock.push_back(child);
                }
            }

            // Now lock everything in sorted order
            lock_node_list(to_lock);

            // Verify no new children slipped in
            size_t expected = 1 + node->children.size();
            if (expected == to_lock.size()) {
                return;  // success
            }

            // Unlock and retry
            for (auto& n : to_lock) {
                n->lock.unlock();
            }
        }
    }

    /**
     * v1TODO: make this and unlock_dnode one function
     */
    void ThtsPool::unlock_cnode_and_children(shared_ptr<ThtsCNode> node) {
        node->lock.unlock();
        for (auto [_obs, child] : node->children)
        {
            child->lock.unlock();
        }
    }

    /**
     * Setter for root node, so thread pool can be reused
    */
    void ThtsPool::reset(
        shared_ptr<ThtsManager> new_thts_manager, 
        shared_ptr<ThtsDNode> new_root_node,
        shared_ptr<ThtsLogger> new_logger) 
    {
        if (work_left()) {
            throw runtime_error("Tried to change root node in thts pool while it was working.");
        }
        thts_manager = new_thts_manager;
        root_node = new_root_node;
        logger = new_logger;
    }

    /**
     * Trial should be ended when we reach a leaf node for the search (that is, it is a leaf in the thts_env, or, it is 
     * at the maximum depth).
     * 
     * Additionally when running in mcts mode, we want to end a trial once a new node has been made.
     */
    bool ThtsPool::should_continue_selection_phase(
        shared_ptr<ThtsDNode> cur_node, bool new_decision_node_created_this_trial) 
    {
        if (cur_node->is_leaf()) return false;
        // if (cur_node->is_sink()) return false;
        // if (cur_node->decision_depth >= thts_manager->max_depth) return false;
        if (thts_manager->mcts_mode && new_decision_node_created_this_trial) return false;
        return true;
    }

    /**
     * Used by a worker thread to run the selection phase of a trial.
     * 
     * Performs the following while 'should_continue_selection_phase' evaluates to true:
     * - decision node visit
     * - decision node select action
     * - chance node visit
     * - chance node sample outcome
     * - add nodes and rewards to 'nodes_to_backup' and 'rewards' vectors
     * 
     * Note that we don't lock cur_node when checking 'should_continue_selection_phase' because it only accesses values 
     * that should remain constant throughout the run.
     * 
     * Throughout the function, whenever a decision/chance node is being used in a function/has function being called, 
     * it is protected by its appropriate lock.
     * 
     * At the end, to make the list of rewards sum to the total return of the trial (consider when the heuristic_fn is 
     * a rollout), we also add the heuristic_value of the last node considered this trial.
     */
    void ThtsPool::run_selection_phase(
        vector<pair<shared_ptr<ThtsDNode>,shared_ptr<ThtsCNode>>>& nodes_to_backup, 
        vector<double>& rewards, 
        ThtsContext& context,
        int tid)
    {
        bool new_decision_node_created_this_trial = false;
        shared_ptr<ThtsDNode> cur_node = root_node;

        while (should_continue_selection_phase(cur_node, new_decision_node_created_this_trial)) {
            // dnode visit + select action
            lock_dnode_and_children(cur_node);
            cur_node->visit_itfc(context);
            shared_ptr<const Action> action = cur_node->select_action_itfc(context);
            shared_ptr<ThtsCNode> chance_node = cur_node->get_child_node_itfc(action);
            unlock_dnode_and_children(cur_node);
            
            // cnode visit + sample outcome
            lock_cnode_and_children(chance_node);
            int pre_visit_children = chance_node->get_num_children();
            chance_node->visit_itfc(context);
            shared_ptr<const Observation> observation = chance_node->sample_observation_itfc(context);
            chance_node->update_empirical_distribution(observation);
            int post_visit_children = chance_node->get_num_children();
            if (post_visit_children > pre_visit_children) {
                new_decision_node_created_this_trial = true;
            }
            shared_ptr<ThtsDNode> decision_node = chance_node->get_child_node_itfc(observation);
            unlock_cnode_and_children(chance_node);

            // push onto 'nodes_to_backup' and 'rewards'
            shared_ptr<const State> state = cur_node->state;
            double reward = thts_manager->thts_env(tid)->get_reward_itfc(state, action, context);
            nodes_to_backup.push_back(make_pair(cur_node, chance_node));
            rewards.push_back(reward);

            cur_node = decision_node;
        }

        // visit the final node and add heuristic value to list of rewards at end
        lock_dnode_and_children(cur_node);
        cur_node->visit_itfc(context);
        rewards.push_back(cur_node->heuristic_value);
        unlock_dnode_and_children(cur_node);
    }


    /**
     * Used by a worker thread to run the backup phase of a trial.
     * 
     * Iterates through the nodes that are needed to be backed up, computes the vectors and sums of rewards that the 
     * backup function expects, and calls backup on each of them. Noting that we are backing up from the bottom, so 
     * we call backup on chance node before decision node.
     * 
     * If have trajectory s0,a0,r0,s1,a1,r1,...,sn,an,rn,hv, where hv=heuristic 
     * value, then, we would pass the backup function for decision node (si) 
     * and chance_node (si,ai) in the form:
     * - rewards_after = [hv,rn,r(n-1),...,ri]
     * - rewards_before = [r0,r1,r2,...,r(i-1)]
     * - total_return_after = sum(rewards_after)
     * - total_return = sum(rewards_after) + sum(rewards_before)
     */
    void ThtsPool::run_backup_phase(
        vector<pair<shared_ptr<ThtsDNode>,shared_ptr<ThtsCNode>>>& nodes_to_backup, 
        vector<double>& rewards, 
        ThtsContext& context)
    {
        double total_return = 0.0;
        for (double& reward : rewards) total_return += reward;

        vector<double> rewards_after;
        vector<double> rewards_before(rewards);

        double heuristic_val_at_end = rewards_before.back();
        rewards_before.pop_back();
        rewards_after.push_back(heuristic_val_at_end);

        double total_return_after = heuristic_val_at_end;

        while (nodes_to_backup.size() > 0) {
            double reward = rewards_before.back();
            rewards_before.pop_back();
            rewards_after.push_back(reward);
            total_return_after += reward;

            pair<shared_ptr<ThtsDNode>,shared_ptr<ThtsCNode>> pr = nodes_to_backup.back();
            shared_ptr<ThtsDNode> decision_node = pr.first;
            shared_ptr<ThtsCNode> chance_node = pr.second;
            nodes_to_backup.pop_back();

            lock_cnode_and_children(chance_node);
            chance_node->backup_itfc(rewards_before, rewards_after, total_return_after, total_return, context);
            unlock_cnode_and_children(chance_node);

            lock_dnode_and_children(decision_node);
            decision_node->backup_itfc(rewards_before, rewards_after, total_return_after, total_return, context);
            unlock_dnode_and_children(decision_node);
        }
    }

    /**
     * Returns if another trial can be run by a worker
     * 
     * If the following are true we can run another trial:
     * - trials_remaining > 0
     * - elapsed_time is within the max_run_time
     * 
     * Checks if each condition is violated in turn, and returns false if violated, otherwise returns true at end.
     */
    bool ThtsPool::work_left() {
        if (trials_remaining <= 0) return false;

        chrono::time_point<chrono::system_clock> cur_time = chrono::system_clock::now();
        chrono::duration<double> elapsed_time = cur_time - start_time;
        if (elapsed_time >= max_run_time) return false;

        return true;
    }

    /**
     * Used by a worker thread to run a single trial of thts
     * 
     * Selection phase uses visit and selection functions to fill 'nodes_to_backup' and 'rewards', passed by ref.
     * Backup phase calls backup on 'nodes_to_backup' passing them rewards from 'rewards'.
     * 
     * Trys to perform logging at the end of this trial (if there is a logger and is time to log)
     */
    void ThtsPool::run_thts_trial(int trials_remaining, int tid) {
        vector<pair<shared_ptr<ThtsDNode>,shared_ptr<ThtsCNode>>> nodes_to_backup;
        vector<double> rewards; 
        
        thts_manager->thts_env(tid)->reset_itfc();
        shared_ptr<ThtsContext> context = 
            thts_manager->thts_env(tid)->sample_context_itfc(tid, *thts_manager);
        thts_manager->register_thts_context(tid,context);
        run_selection_phase(nodes_to_backup, rewards, *context, tid);
        run_backup_phase(nodes_to_backup, rewards, *context);

        try_log();
    }
    
    /**
    * If logging, grabs logging lock, checks if it is time to log and calls 'log' if it is, making sure to grab the
    * lock for the root node as 'log' doesn't do that but needs to access the root node. When all trials are completed
    * (trials_completed == num_trials), then call 'update_prior_runtime' for logger as it should be called after all
    * trials are finished.
    */
    void ThtsPool::try_log() {
        if (logger != nullptr) {
            lock_guard<mutex> logging_lg(logging_lock);
            trials_completed++;
            logger->trial_completed();

            if (logger->should_log()) {
                lock_guard<mutex> root_node_lg(root_node->lock);
                logger->log(root_node);
            }

            if (trials_completed == num_trials) {
                logger->update_prior_runtime();
            }
        }
    }

    /**
     * The worker thread function.
     * 
     * - first waiot
     * 
     * while this thts pool is 'alive' work threads perform the following (description of code from top to bottom):
     * - indicate that they have completed a trial/unit of work, by decrementing num_threads_working
     * - if can't work (run a trial): 
     *      - notify work_left_cv to try to signal any threads waiting on workers completing
     *      - wait on work_left_cv until notified
     * - if just woken up from waiting on work_left_cv: 
     *      -check if thread_pool is still alive, and if not, exit
     * - when can run a trial: 
     *      - decrement trials remaining to indicate we're going to run a trial
     *      - increment num_threads_working to indicate we're starting some work
     *      - run trial
     * 
     * A lock_guard is used to make sure that the work_left_lock is locked throughout the routine, except while waiting 
     * on work_left_cv, and when unlocked around running a trial.
     */
    void ThtsPool::worker_fn(int tid) {
        // setup thread
        // TODO: when integrate with main branch, don't want this necessarily here
        //          probably re-introduce PyThtsPool (in py/py_thts.{h,cpp}), which is currently commented out
        // TODO: also re-introduce MoPyThtsPool (in py/mo_py_thts.{h,cpp}) in MO branch
        thts_manager->register_thread_id(tid);

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

    /**
     * Waits on workers to complete their work
     */
    void ThtsPool::join() {
        lock_guard<mutex> lg(work_left_lock);
        while (work_left() || num_threads_working > 0) {
            work_left_cv.wait(work_left_lock);
        }
    }

    /**
     * Instructs the ThtsPool to start running trials
     * 
     * This function performs the following steps (ignoring logging):
     * - grabs the work_left_lock
     * - initialises variables used by 'work_left' function, to indicate how many trials/how long to run thts for
     * - unlocks work_left_lock (join re-aquires the work_left_lock)
     * - signals worker threads via work_left_cv to start working
     * - if blocking, calls join to wait 
     * 
     * For logging we call the function that needs to be called at the start of a 'run_trials' call, so the logger 
     * knows the start time. And if the logger is empty, it adds an origin point/entry.
     * 
     * Args:
     *      max_trials: The maximum number of trials to run
     *      max_time: The maximum duration (in seconds) to run trials for
     *      blocking: If this call is blocking, and waits for the trials to be completed
     */
    void ThtsPool::run_trials(int max_trials, double max_time, bool blocking) {
        if (logger != nullptr) {
            lock_guard<mutex> lg(logging_lock);
            if (logger->size() == 0) {
                logger->add_origin_entry();
            }
            logger->reset_start_time();
        }

        work_left_lock.lock();
        num_trials = max_trials;
        trials_remaining = max_trials;
        start_time = std::chrono::system_clock::now();
        max_run_time =  std::chrono::duration<double>(max_time);
        work_left_lock.unlock();

        work_left_cv.notify_all();
        if (blocking) join();
    }
}