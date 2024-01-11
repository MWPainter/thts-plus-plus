#pragma once

#include "thts_manager.h"
#include "thts_types.h"

#include <Eigen/Dense>

namespace thts::helper {
    /**
     * A default heuristic function that returns a constant zero vector
     */
    template <unsigned int dim>
    Eigen::ArrayXd mo_zero_heuristic_fn(std::shared_ptr<const State> state, std::shared_ptr<ThtsEnv> env);
}

#include "mo/mo_helper_templates.cc"