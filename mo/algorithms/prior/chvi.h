#pragma once

#include "mo/mo_thts_types.h"
#include "mo/data_structures/convex_hull.h"

namespace thts {

    typedef std::unordered_set<std::shared_ptr<State>> StateSet;
    typedef std::unordered_map<std::shared_ptr<State>, 
                               std::unordered_map<std::shared_ptr<Action>, 
                                            std::unordered_map<std::shared_ptr<State>, double>>> TransitionProbs;
    typedef std::unordered_map<std::shared_ptr<State>, 
                               std::unordered_map<std::shared_ptr<Action>, Vec>> RewardMap;
    typedef std::unordered_map<std::shared_ptr<State>, std::shared_ptr<ConvexHull>> VMap;
    
    class Chvi {
        int num_threads;
        double max_time;

        const int dim;
        const std::shared_ptr<State> start_state;
        const StateSet states;
        const StateSet sink_states;
        const TransitionProbs transition_probs;
        const RewardMap reward_map;
        
        // Double buffering for thread-safe synchronous value iteration
        VMap chvi_values;           // Values being read from (previous iteration)
        VMap chvi_values_next;      // Values being written to (current iteration)

        public:
            Chvi(
                int num_threads, 
                double max_time, 
                int dim,
                std::shared_ptr<State> start_state, 
                StateSet states, 
                StateSet sink_states, 
                TransitionProbs transition_probs, 
                RewardMap reward_map);
            virtual ~Chvi() = default;

            void run();

            ConvexHull get_chvi_value(std::shared_ptr<State> state) const;
            ConvexHull get_root_chvi_value() const;

        private:
            void backup(std::shared_ptr<State> state);
    };
}