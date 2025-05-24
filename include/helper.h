#pragma once

#include "thts_types.h"

#include <memory>

// forward declares
namespace thts {
    class ThtsEnv;
    class ThtsManager;
}

namespace thts::helper {
    /**
     * A default heuristic function that returns a constant zero
     */
    double zero_heuristic_fn(
        std::shared_ptr<const State> state, ThtsEnv& env, ThtsManager& manager, int depth=0);

    /**
     * The rollout heuristic function, that returns an MC estimate of 'state' with a rollout with random policy
     */
    double rollout_heuristic_fn(
        std::shared_ptr<const State> state, ThtsEnv& env, ThtsManager& manager, int depth);

    /**
     * String split function
     */
    std::vector<std::string> string_split(const std::string& s, const std::string& delimiter=",");
}