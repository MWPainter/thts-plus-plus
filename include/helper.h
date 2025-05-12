#pragma once

#include "thts_types.h"

#include <memory>

// forward declar ThtsEnv
namespace thts {
    class ThtsEnv;
}

namespace thts::helper {
    /**
     * A default heuristic function that returns a constant zero
     */
    double zero_heuristic_fn(std::shared_ptr<const State> state, std::shared_ptr<ThtsEnv> env=nullptr);

    /**
     * String split function
     */
    std::vector<std::string> string_split(const std::string& s, const std::string& delimiter=",");
}