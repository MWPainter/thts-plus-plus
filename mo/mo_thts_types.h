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
     * An implementation of state containing a vector of integers as the state.
     */
    class IntVectorState : public State {
        public:
            std::vector<int> state;

            IntVectorState(std::vector<int>& v) : state(v) {}
            virtual ~IntVectorState() = default;
            virtual std::size_t hash() const override;
            bool equals(const IntVectorState& other) const;
            virtual bool equals_itfc(const Observation& other) const override;
            virtual std::string get_pretty_print_string() const override;
    };


    typedef std::unordered_map<std::shared_ptr<const IntVectorState>,double> IntVectorStateDistr;



    /**
     * Typedef for heuristic function pointers
     * First used in thts_decision_node.h and thts_chance_node.h
     * N.B. The & here is to get address as we want function pointers
     */  
    Eigen::ArrayXd _DummyMoHeuristicFn(std::shared_ptr<const State> s, std::shared_ptr<ThtsEnv> env);
    typedef decltype(&_DummyMoHeuristicFn) MoHeuristicFnPtr;
}


namespace std {
    using namespace thts;


    ostream& operator<<(ostream& os, const IntVectorState& state);
    ostream& operator<<(ostream& os, const shared_ptr<const IntVectorState>& state);

}