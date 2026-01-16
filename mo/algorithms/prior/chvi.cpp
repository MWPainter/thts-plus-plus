#include "mo/algorithms/prior/chvi.h"

#include "mo/mo_thts_context.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

#include <iostream>

namespace thts {
    using namespace std;

    /**
     * Constructor - initializes local variables and chvi_values
     */
    Chvi::Chvi(
        int num_threads,
        int dim,
        shared_ptr<const State> start_state,
        StateSet states,
        StateSet sink_states,
        TransitionProbs transition_probs,
        RewardMap reward_map,
        int convex_hull_max_size,
        double convex_hull_tolerance) :
            num_threads(num_threads),
            dim(dim),
            convex_hull_max_size(convex_hull_max_size),
            convex_hull_tolerance(convex_hull_tolerance),
            start_state(std::move(start_state)),
            states(std::move(states)),
            sink_states(std::move(sink_states)),
            transition_probs(std::move(transition_probs)),
            reward_map(std::move(reward_map)),
            chvi_values(),
            chvi_values_next(),
            chvi_q_values(),
            chvi_q_values_next(),
            work_queue(),
            queue_mutex(),
            queue_cv(),
            should_stop(false),
            threads_waiting(0),
            non_sink_states(),
            completed_iterations(-1),
            backups_completed_current(0),
            total_backups_per_iter(0)
    {
        // Initialize chvi_values with zero vector convex hull for each state
        Vec zero_vec(dim, 0.0);
        for (const auto& state : this->states) {
            chvi_values[state] = make_shared<ConvexHull>(zero_vec, convex_hull_max_size, convex_hull_tolerance);
            chvi_values_next[state] = make_shared<ConvexHull>(zero_vec, convex_hull_max_size, convex_hull_tolerance);
        }
        
        // Initialize Q values for each state-action pair
        for (const auto& [state, action_map] : this->transition_probs) {
            for (const auto& [action, _] : action_map) {
                chvi_q_values[state][action] = make_shared<ConvexHull>(zero_vec, convex_hull_max_size, convex_hull_tolerance);
                chvi_q_values_next[state][action] = make_shared<ConvexHull>(zero_vec, convex_hull_max_size, convex_hull_tolerance);
            }
        }
        
        // Build list of non-sink states (cached for reuse)
        for (const auto& state : this->states) {
            if (!this->sink_states.contains(state)) {
                non_sink_states.push_back(state);
            }
        }
        
        // Set total backups per iteration
        total_backups_per_iter = non_sink_states.size();
    }

    /**
     * Perform a single Bellman backup on a state using ConvexHull value iteration
     * 
     * V(s) = union_a [ R(s,a) + sum_{s'} P(s'|s,a) * V(s') ]
     * 
     * Where the union creates the convex hull of all action values
     */
    void Chvi::backup(shared_ptr<const State> state) {
        // Sink states have zero value (terminal)
        if (sink_states.contains(state)) {
            return;
        }

        // Check if this state has any actions
        auto state_it = transition_probs.find(state);
        if (state_it == transition_probs.end()) {
            return;  // No actions available from this state
        }

        const auto& action_map = state_it->second;
        
        // Compute the value for each action and union them
        ConvexHull new_value;
        bool first_action = true;

        for (const auto& [action, next_state_probs] : action_map) {
            // Get reward for this state-action pair
            Vec reward = reward_map.at(state).at(action);

            // Compute expected next state value: sum_{s'} P(s'|s,a) * V(s')
            ConvexHull expected_next_value = ConvexHull(Vec(dim, 0.0), convex_hull_max_size, convex_hull_tolerance);

            for (const auto& [next_state, prob] : next_state_probs) {
                // Get the value of the next state
                ConvexHull next_value = *chvi_values.at(next_state);
                
                // Scale by transition probability
                ConvexHull scaled_next_value = next_value.scale(prob);
                
                // Minkowski sum for expectation
                expected_next_value = expected_next_value.add(scaled_next_value);
            }

            // Q(s,a) = R(s,a) + expected_next_value
            ConvexHull action_value = expected_next_value.add(reward);
            
            // Store Q value for this state-action pair
            *chvi_q_values_next[state][action] = action_value;

            // Union over actions
            if (first_action) {
                new_value = action_value;
                first_action = false;
            } else {
                new_value = new_value.combine(action_value);
            }
        }

        // Update the value for this state (write to next buffer)
        if (!first_action) {
            *chvi_values_next[state] = new_value;
        }
    }

    /**
     * Multi-threaded value iteration
     * 
     * Uses a work queue that threads pull states from. An orchestrator thread
     * refills the queue with non-sink states for each iteration.
     * 
     * Stops when:
     * - max_time seconds have elapsed, OR
     * - max_iter iterations have been completed
     * 
     * Time is checked between backups, so long-running iterations won't prevent timely stopping.
     * 
     * Can be called multiple times to resume computation from where it left off.
     */
    void Chvi::run(double max_time, int max_iter) {
        // Reset stop flags for new run
        should_stop.store(false);
        threads_waiting.store(0);
        
        // Time tracking - shared across threads
        auto start_time_point = chrono::steady_clock::now();
        
        // Helper lambda to check if time limit exceeded
        // Capture start_time_point and max_time by value to ensure thread safety
        auto time_exceeded = [start_time_point, max_time]() -> bool {
            auto current_time = chrono::steady_clock::now();
            double elapsed = chrono::duration<double>(current_time - start_time_point).count();
            return elapsed >= max_time;
        };

        // Worker thread function
        auto worker_fn = [this, time_exceeded](int thread_id) {
            while (!this->should_stop.load()) {
                shared_ptr<const State> state_to_backup;
                
                {
                    unique_lock<mutex> lock(this->queue_mutex);
                    
                    // If queue is empty, wait for work or stop signal
                    if (this->work_queue.empty()) {
                        this->threads_waiting++;
                        this->queue_cv.notify_all();  // Notify orchestrator that we're waiting
                        
                        // Wait for work or stop signal using cv wait
                        // Checking predicate / wake up is atomic
                        // (Important because protecting variables with atomic types, rather than mutex)
                        this->queue_cv.wait(lock, [this]() {
                            return !this->work_queue.empty() || this->should_stop.load();
                        });
                        
                        this->threads_waiting--;
                    }
                    
                    // Check if we have been signalled to stop
                    if (this->should_stop.load()) {
                        return;
                    }
                
                    // Check time limit before performing backup (and pulling work from queue)
                    if (time_exceeded()) {
                        this->should_stop.store(true);
                        this->queue_cv.notify_all();
                        return;
                    }
                    
                    // Check if queue has work (might be empty after spurious wakeup)
                    if (!this->work_queue.empty()) {
                        state_to_backup = this->work_queue.front();
                        this->work_queue.pop();
                    } else {
                        // Spurious wakeup or race, loop again
                        continue;
                    }
                }
                
                // Perform backup outside of lock
                if (state_to_backup) {
                    backup(state_to_backup);
                    // Increment backup counter (thread-safe)
                    this->backups_completed_current++;
                }
                
                // Check time limit after backup
                if (time_exceeded()) {
                    this->should_stop.store(true);
                    this->queue_cv.notify_all();
                    return;
                }
            }
        };

        // Start worker threads
        vector<thread> workers;
        workers.reserve(num_threads);
        for (int i = 0; i < num_threads; i++) {
            workers.emplace_back(worker_fn, i);
        }

        // Orchestrator loop
        int iteration = 0;
        
        while (iteration++ < max_iter && !time_exceeded() && !should_stop.load()) {
            // Fill the queue with states for this iteration (only if queue is empty)
            {
                lock_guard<mutex> lock(queue_mutex);
                if (work_queue.empty()) {
                    // Starting a new iteration - update iteration counters
                    completed_iterations++;
                    backups_completed_current.store(0);
                    for (const auto& state : non_sink_states) {
                        work_queue.push(state);
                    }
                }
            }
            queue_cv.notify_all();

            // Wait for all workers to finish processing the queue or stop signal
            unique_lock<mutex> lock(queue_mutex);
            queue_cv.wait(lock, [this]() {
                return this->should_stop.load() || 
                        (this->work_queue.empty() && this->threads_waiting.load() == this->num_threads);
            });

            // Check if time limit exceeded
            if (time_exceeded() || should_stop.load()) {
                break;
            }
            
            // Check if completed iteration 
            if (this->work_queue.empty()) {
                // Swap buffers: copy next values to current values for next iteration
                for (const auto& state : non_sink_states) {
                    *chvi_values[state] = *chvi_values_next[state];
                }
                
                // Swap Q value buffers
                for (const auto& [state, action_map] : chvi_q_values_next) {
                    for (const auto& [action, _] : action_map) {
                        *chvi_q_values[state][action] = *chvi_q_values_next[state][action];
                    }
                }
            }
        }

        // Signal workers to stop and wake them up
        should_stop.store(true);
        queue_cv.notify_all();

        // Wait for all workers to finish
        for (auto& worker : workers) {
            worker.join();
        }
        
        // Final copy: ensure chvi_values and Q values have the latest results
        for (const auto& state : non_sink_states) {
            *chvi_values[state] = *chvi_values_next[state];
        }
        
        for (const auto& [state, action_map] : chvi_q_values_next) {
            for (const auto& [action, _] : action_map) {
                *chvi_q_values[state][action] = *chvi_q_values_next[state][action];
            }
        }
    }

    /**
     * Get the CHVI value for a given state
     */
    ConvexHull Chvi::get_chvi_value(shared_ptr<const State> state) const {
        auto it = chvi_values.find(state);
        if (it != chvi_values.end()) {
            return *it->second;
        }
        // Return empty convex hull if state not found
        return ConvexHull(convex_hull_max_size, convex_hull_tolerance);
    }

    /**
     * Get the CHVI value for the start state
     */
    ConvexHull Chvi::get_root_chvi_value() const {
        return get_chvi_value(start_state);
    }

    /**
     * Get the CHVI Q-value for a given state-action pair
     */
    ConvexHull Chvi::get_chvi_q_value(shared_ptr<const State> state, shared_ptr<const Action> action) const {
        auto state_it = chvi_q_values.find(state);
        if (state_it != chvi_q_values.end()) {
            auto action_it = state_it->second.find(action);
            if (action_it != state_it->second.end()) {
                return *action_it->second;
            }
        }
        // Return empty convex hull if state-action pair not found
        return ConvexHull(convex_hull_max_size, convex_hull_tolerance);
    }

    /**
     * Get the number of iterations run (including partial progress)
     * 
     * Returns completed_iterations + (backups_completed_current / total_backups_per_iter)
     * 
     * Example: If on the second iteration and 50 out of 100 backups have been completed,
     *          returns 1.0 + (50 / 100) = 1.5
     */
    double Chvi::get_num_iters_run() const {
        int completed = completed_iterations.load();
        int current_backups = backups_completed_current.load();
        
        if (total_backups_per_iter == 0) {
            return static_cast<double>(completed);
        }
        
        double partial_iteration = static_cast<double>(current_backups) / static_cast<double>(total_backups_per_iter);
        return static_cast<double>(completed) + partial_iteration;
    }

    /**
     * CHVI Eval Policy Implementation
     */

    /**
     * Constructor - takes a shared pointer to Chvi object
     * Note: thts_env and manager should be set via clone() or directly
     */
    ChviEvalPolicy::ChviEvalPolicy(
        shared_ptr<Chvi> chvi, 
        shared_ptr<ThtsEnv> thts_env, 
        shared_ptr<ThtsManager> manager) :
            EvalPolicy(nullptr, thts_env, manager),
            chvi(chvi)
    {
        // thts_env and manager will be set when cloned or used
    }

    /**
     * Virtual copy constructor
     */
    shared_ptr<EvalPolicy> ChviEvalPolicy::clone(shared_ptr<ThtsEnv> thts_env) {
        // Create a new ChviEvalPolicy with the same chvi object
        return make_shared<ChviEvalPolicy>(this->chvi, thts_env, this->manager);
    }

    /**
     * Reset - no-op for CHVI policy (no state to reset)
     */
    void ChviEvalPolicy::reset() {
        // No state to reset for CHVI policy
    }

    /**
     * Get action using CHVI Q values
     * 
     * Selects the action that maximizes the linear utility with the context weight:
     * argmax_a Q(s,a) · w
     */
    shared_ptr<const Action> ChviEvalPolicy::get_action(
        shared_ptr<const State> state, ThtsContext& context) 
    {
        // Cast context to MoThtsContext to get context weight
        MoThtsContext* mo_context = dynamic_cast<MoThtsContext*>(&context);
        Vec context_weight = mo_context->context_weight;

        // Find action with maximum linear utility
        shared_ptr<ActionVector> actions = thts_env->get_valid_actions_itfc(state, context);
        unordered_map<shared_ptr<const Action>, double> action_utilities;
        double max_utility = numeric_limits<double>::lowest();
        
        for (shared_ptr<const Action> action : *actions) {
            // Get Q value for this state-action pair
            ConvexHull q_value = chvi->get_chvi_q_value(state, action);
            
            // Compute linear utility: Q(s,a) · w
            double utility = q_value.get_max_linear_utility(context_weight);
            action_utilities[action] = utility;
            
            if (utility > max_utility) {
                max_utility = utility;
            }
        }

        // Collect all actions with maximum utility (for tie-breaking)
        vector<shared_ptr<const Action>> best_actions;
        for (const auto& [action, utility] : action_utilities) {
            if (utility == max_utility) {
                best_actions.push_back(action);
            }
        }
        
        int indx = manager->get_rand_int(0, best_actions.size());
        return best_actions[indx];
    }

    /**
     * Update step - no-op for CHVI policy (no state to update)
     */
    void ChviEvalPolicy::update_step(
        shared_ptr<const Action> action, 
        shared_ptr<const Observation> obsv) 
    {
        // No state to update for CHVI policy
    }
}

