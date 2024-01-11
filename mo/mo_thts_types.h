#pragma once

#include "thts_types.h"

#include <memory>
#include <Eigen/Dense>

/**
 * thts_types.h
 * 
 * This file contains some base types for multi objective algorithms
 * 
 * For now this should just be some typedefs
 */

namespace thts {
    /**
     * Typedef for heuristic function pointers
     * First used in thts_decision_node.h and thts_chance_node.h
     * N.B. The & here is to get address as we want function pointers
     */  
    Eigen::ArrayXd _DummyMoHeuristicFn(std::shared_ptr<const State> s, std::shared_ptr<ThtsEnv> env);
    typedef decltype(&_DummyMoHeuristicFn) MoHeuristicFnPtr;
}