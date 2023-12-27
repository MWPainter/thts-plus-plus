#pragma once

#include "thts_types.h"

#include <pybind11/pybind11.h>

/**
 * py_thts_types.h
 * 
 * This file contains wrappers around python objects to be used as States, Actions and Observations.
 */

namespace thts::python {
    // forward declares
    class PyThtsEnv;

    // namespace includes
    namespace py = pybind11;

    /**
     * Python Observations
     * Wrapper around an arbitrary python object
    */
    class PyObservation : public Observation {
        friend PyThtsEnv;

        protected:
            py::object py_obs;
        
        public:
            PyObservation(py::object obs);
            virtual ~PyObservation() = default;
            bool equals(const PyObservation& other) const;

            virtual std::size_t hash() const;
            virtual bool equals_itfc(const Observation& other) const;
            virtual std::string get_pretty_print_string() const;
    };

    /**
     * Python State
     * Wrapper around an arbitrary python object
    */
    class PyState : public State {
        friend PyThtsEnv;

        protected:
            py::object py_state;
        
        public:
            PyState(py::object state);
            virtual ~PyState() = default;
            bool equals(const PyState& other) const;
            
            virtual std::size_t hash() const;
            virtual bool equals_itfc(const State& other) const;
            virtual std::string get_pretty_print_string() const;
    };

    /**
     * Python Action
     * Wrapper around an arbitrary python object
    */
    class PyAction : public Action {
        friend PyThtsEnv;
        
        protected:
            py::object py_action;
        
        public:
            PyAction(py::object action);
            virtual ~PyAction() = default;
            bool equals(const PyAction& other) const;

            virtual std::size_t hash() const;
            virtual bool equals_itfc(const Action& other) const;
            virtual std::string get_pretty_print_string() const;
    };
}

/**
 * Forward declaring the hash, equality and output stream functions defined in thts_types.cpp.
 * Needed so other files know to look at thts_types.o to find implementations of these functions.
 */
namespace std {
    using namespace thts::python;

    /**
     * Hash, equality class and output stream function definitions for PyObservation.
     */
    template <> 
    struct hash<PyObservation> {
        size_t operator()(const PyObservation&) const;
    };

    template <> 
    struct hash<shared_ptr<const PyObservation>> {
        size_t operator()(const shared_ptr<const PyObservation>&) const;
    };
    
    bool operator==(const PyObservation& lhs, const PyObservation& rhs);
    bool operator==(const shared_ptr<const PyObservation>& lhs, const shared_ptr<const PyObservation>& rhs);

    template <> 
    struct equal_to<PyObservation> {
        bool operator()(const PyObservation&, const PyObservation&) const;
    };

    template <> 
    struct equal_to<shared_ptr<const PyObservation>> {
        bool operator()(const shared_ptr<const PyObservation>&, const shared_ptr<const PyObservation>&) const;
    };

    ostream& operator<<(ostream& os, const PyObservation& observation);
    ostream& operator<<(ostream& os, const shared_ptr<const PyObservation>& observation);

    /**
     * Hash, equality class and output stream function definitions for PyState.
     */
    template <> 
    struct hash<PyState> {
        size_t operator()(const PyState&) const;
    };

    template <> 
    struct hash<std::shared_ptr<const PyState>> {
        size_t operator()(const shared_ptr<const PyState>&) const;
    };
    
    bool operator==(const PyState& lhs, const PyState& rhs);
    bool operator==(const shared_ptr<const PyState>& lhs, const shared_ptr<const PyState>& rhs);

    template <> 
    struct equal_to<PyState> {
        bool operator()(const PyState&, const PyState&) const;
    };

    template <> 
    struct equal_to<shared_ptr<const PyState>> {
        bool operator()(const shared_ptr<const PyState>&, const shared_ptr<const PyState>&) const;
    };

    ostream& operator<<(ostream& os, const PyState& state);
    ostream& operator<<(ostream& os, const shared_ptr<const PyState>& state);

    /**
     * Hash, equality class and output stream function definitions for PyAction.
     */
    template <> 
    struct hash<PyAction> {
        size_t operator()(const PyAction&) const;
    };

    template <> 
    struct hash<shared_ptr<const PyAction>> {
        size_t operator()(const shared_ptr<const PyAction>&) const;
    };
    
    bool operator==(const PyAction& lhs, const PyAction& rhs);
    bool operator==(const shared_ptr<const PyAction>& lhs, const shared_ptr<const PyAction>& rhs);

    template <> struct equal_to<PyAction> {
        bool operator()(const PyAction&, const PyAction&) const;
    };

    template <> struct equal_to<shared_ptr<const PyAction>> {
        bool operator()(const shared_ptr<const PyAction>&, const shared_ptr<const PyAction>&) const;
    };

    ostream& operator<<(ostream& os, const PyAction& action);
    ostream& operator<<(ostream& os, const shared_ptr<const PyAction>& action);
}