#pragma once

#include "thts_types.h"

#include <memory>

namespace thts::helper {
    /**
     * A default heuristic function that returns a constant zero
     */
    double zero_heuristic_fn(std::shared_ptr<const State> state, std::shared_ptr<const Action> action=nullptr);
}