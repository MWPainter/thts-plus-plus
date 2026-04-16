#pragma once

#include "mo/mo_thts_types.h"
#include "mo/data_structures/convex_hull.h"

#include "mc_eval.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>

namespace thts {

    typedef std::unordered_set<std::shared_ptr<const State>> StateSet;
    typedef std::vector<std::shared_ptr<const State>> OrderedStateVec;
    typedef std::unordered_map<std::shared_ptr<const State>, 
                               std::unordered_map<std::shared_ptr<const Action>, 
                                            std::unordered_map<std::shared_ptr<const State>, double>>> TransitionProbs;
    typedef std::unordered_map<std::shared_ptr<const State>, 
                               std::unordered_map<std::shared_ptr<const Action>, Vec>> RewardMap;
    typedef std::unordered_map<std::shared_ptr<const State>, std::shared_ptr<ConvexHull>> VMap;
    typedef std::unordered_map<std::shared_ptr<const State>, 
                               std::unordered_map<std::shared_ptr<const Action>, std::shared_ptr<ConvexHull>>> QMap;
    
    class Chvi {
        protected:
            int num_threads;

            const int dim;
            const int convex_hull_max_size;
            const double convex_hull_tolerance;
            const std::shared_ptr<const State> start_state;
            const StateSet states;
            const StateSet sink_states;
            const TransitionProbs transition_probs;
            const RewardMap reward_map;
            
            // Double buffering for thread-safe synchronous value iteration
            VMap chvi_values;           // Values being read from (previous iteration)
            VMap chvi_values_next;      // Values being written to (current iteration)
            
            // Q-value storage: Q(s,a) for each state-action pair
            QMap chvi_q_values;         // Q values from previous iteration
            QMap chvi_q_values_next;    // Q values being written to (current iteration)
            
            // Work queue and synchronization for multi-threaded execution
            std::queue<std::shared_ptr<const State>> work_queue;
            std::mutex queue_mutex;
            std::condition_variable queue_cv;
            std::atomic<bool> should_stop;
            std::atomic<int> threads_waiting;

            // Cached list of non-sink states (protected so subclasses can control ordering)
            std::vector<std::shared_ptr<const State>> non_sink_states;
            
            // When true, backup reads from chvi_values_next (Gauss-Seidel style) instead of
            // chvi_values (Jacobi style), so ordered sweeps see freshly computed values.
            bool use_gauss_seidel = false;

        private:
            // Iteration tracking
            std::atomic<int> completed_iterations;      // Number of fully completed iterations
            std::atomic<int> backups_completed_current; // Number of backups completed in current iteration
            int total_backups_per_iter;                 // Total number of backups per iteration (size of non_sink_states)
            std::atomic<int> total_backups_completed;                // Total number of backups completed

        public:
            Chvi(
                int num_threads, 
                int dim,
                std::shared_ptr<const State> start_state, 
                StateSet states, 
                StateSet sink_states, 
                TransitionProbs transition_probs, 
                RewardMap reward_map,
                int convex_hull_max_size=-1,
                double convex_hull_tolerance=1e-9);
            virtual ~Chvi() = default;

            void run(double max_time, int max_iter);

            ConvexHull get_chvi_value(std::shared_ptr<const State> state) const;
            ConvexHull get_root_chvi_value() const;
            ConvexHull get_chvi_q_value(std::shared_ptr<const State> state, std::shared_ptr<const Action> action) const;
            
            double get_num_iters_run() const;
            int get_total_backups() const;

        private:
            void backup(std::shared_ptr<const State> state);
    };


    /**
     * ChviOrdered
     * 
     * Subclass of Chvi that takes an ordered vector of states instead of an unordered set.
     * The backup order follows the vector ordering, which is useful for topological or
     * prioritised orderings where convergence depends on the sweep order.
     */
    class ChviOrdered : public Chvi {
        public:
            ChviOrdered(
                int num_threads,
                int dim,
                std::shared_ptr<const State> start_state,
                OrderedStateVec ordered_states,
                OrderedStateVec ordered_sink_states,
                TransitionProbs transition_probs,
                RewardMap reward_map,
                int convex_hull_max_size=-1,
                double convex_hull_tolerance=1e-9,
                bool gauss_seidel=true);
            virtual ~ChviOrdered() = default;

        private:
            static StateSet to_state_set(const OrderedStateVec& vec);
    };


    /**
     * CHVI Eval Policy
     * 
     * Implement an eval policy with CHVI values for the actions.
    */
    class ChviEvalPolicy : public EvalPolicy {
        public:
            std::shared_ptr<Chvi> chvi;

            ChviEvalPolicy(
                std::shared_ptr<Chvi> chvi,
                std::shared_ptr<ThtsEnv> thts_env,
                std::shared_ptr<ThtsManager> manager);
            ~ChviEvalPolicy() = default;

            virtual std::shared_ptr<EvalPolicy> clone(std::shared_ptr<ThtsEnv> thts_env) override;
            virtual void reset() override;
            virtual std::shared_ptr<const Action> get_action(std::shared_ptr<const State> state, ThtsContext& context) override;
            virtual void update_step(std::shared_ptr<const Action> action, std::shared_ptr<const Observation> obsv) override;
    };
}