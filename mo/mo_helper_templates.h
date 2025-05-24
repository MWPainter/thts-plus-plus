#pragma once

#include "thts_manager.h"
#include "thts_types.h"

#include <Eigen/Dense>

// forward declares
namespace thts {
    class MoThtsEnv;
    class MoThtsManager;
}

namespace thts::helper {
    /**
     * A default heuristic function that returns a constant zero vector
     */
    template <unsigned int dim>
    Eigen::ArrayXd mo_zero_heuristic_fn(std::shared_ptr<const State> state, MoThtsEnv& env, MoThtsManager& manager, int depth);
}

#include "mo/mo_helper_templates.cc"