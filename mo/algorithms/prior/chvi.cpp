#include "mo/algorithms/prior/chvi.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace thts {
    using namespace std;

    /**
     * Constructor - initializes local variables and chvi_values
     */
    Chvi::Chvi(
        int num_threads,
        double max_time,
        int dim,
        shared_ptr<State> start_state,
        StateSet states,
        StateSet sink_states,
        TransitionProbs transition_probs,
        RewardMap reward_map) :
            num_threads(num_threads),
            max_time(max_time),
            dim(dim),
            start_state(std::move(start_state)),
            states(std::move(states)),
            sink_states(std::move(sink_states)),
            transition_probs(std::move(transition_probs)),
            reward_map(std::move(reward_map)),
            chvi_values(),
            chvi_values_next()
    {
        // Initialize chvi_values with zero vector convex hull for each state
        Vec zero_vec(dim, 0.0);
        for (const auto& state : this->states) {
            chvi_values[state] = make_shared<ConvexHull>(zero_vec);
            chvi_values_next[state] = make_shared<ConvexHull>(zero_vec);
        }
    }

    /**
     * Perform a single Bellman backup on a state using ConvexHull value iteration
     * 
     * V(s) = union_a [ R(s,a) + sum_{s'} P(s'|s,a) * V(s') ]
     * 
     * Where the union creates the convex hull of all action values
     */
    void Chvi::backup(shared_ptr<State> state) {
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
            ConvexHull expected_next_value = ConvexHull(Vec(dim, 0.0));

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
     */
    void Chvi::run() {
        // Work queue and synchronization primitives
        queue<shared_ptr<State>> work_queue;
        mutex queue_mutex;
        condition_variable queue_cv;
        
        // Control flags
        atomic<bool> should_stop(false);
        atomic<int> threads_waiting(0);
        atomic<bool> iteration_complete(false);

        // Build list of non-sink states for iteration
        vector<shared_ptr<State>> non_sink_states;
        for (const auto& state : states) {
            if (!sink_states.contains(state)) {
                non_sink_states.push_back(state);
            }
        }

        // Worker thread function
        auto worker_fn = [&](int thread_id) {
            while (!should_stop.load()) {
                shared_ptr<State> state_to_backup;
                
                {
                    unique_lock<mutex> lock(queue_mutex);
                    
                    // Wait for work or stop signal
                    while (work_queue.empty() && !should_stop.load() && !iteration_complete.load()) {
                        threads_waiting++;
                        queue_cv.notify_all();  // Notify orchestrator that we're waiting
                        queue_cv.wait(lock);
                        threads_waiting--;
                    }
                    
                    // Check if we should stop
                    if (should_stop.load()) {
                        return;
                    }
                    
                    // Check if queue has work
                    if (!work_queue.empty()) {
                        state_to_backup = work_queue.front();
                        work_queue.pop();
                    } else {
                        // No work and iteration might be complete, wait for next iteration
                        continue;
                    }
                }
                
                // Perform backup outside of lock
                if (state_to_backup) {
                    backup(state_to_backup);
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
        auto start_time_point = chrono::steady_clock::now();
        
        while (true) {
            // Check if we've exceeded max_time
            auto current_time = chrono::steady_clock::now();
            double elapsed = chrono::duration<double>(current_time - start_time_point).count();
            
            if (elapsed >= max_time) {
                break;
            }

            // Fill the queue with states for this iteration
            {
                lock_guard<mutex> lock(queue_mutex);
                iteration_complete.store(false);
                for (const auto& state : non_sink_states) {
                    work_queue.push(state);
                }
            }
            queue_cv.notify_all();

            // Wait for all workers to finish processing the queue
            {
                unique_lock<mutex> lock(queue_mutex);
                queue_cv.wait(lock, [&]() {
                    return work_queue.empty() && threads_waiting.load() == num_threads;
                });
                iteration_complete.store(true);
            }
            
            // Swap buffers: copy next values to current values for next iteration
            for (const auto& state : non_sink_states) {
                *chvi_values[state] = *chvi_values_next[state];
            }
        }

        // Signal workers to stop and wake them up
        should_stop.store(true);
        queue_cv.notify_all();

        // Wait for all workers to finish
        for (auto& worker : workers) {
            worker.join();
        }
        
        // Final copy: ensure chvi_values has the latest results
        for (const auto& state : non_sink_states) {
            *chvi_values[state] = *chvi_values_next[state];
        }
    }

    /**
     * Get the CHVI value for a given state
     */
    ConvexHull Chvi::get_chvi_value(shared_ptr<State> state) const {
        auto it = chvi_values.find(state);
        if (it != chvi_values.end()) {
            return *it->second;
        }
        // Return empty convex hull if state not found
        return ConvexHull();
    }

    /**
     * Get the CHVI value for the start state
     */
    ConvexHull Chvi::get_root_chvi_value() const {
        return get_chvi_value(start_state);
    }
}

