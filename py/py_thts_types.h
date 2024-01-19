#pragma once

#include "thts_types.h"
#include "py/pickle_wrapper.h"

#include <pybind11/pybind11.h>

#include <memory>
#include <mutex>

/**
 * py_thts_types.h
 * 
 * This file contains wrappers around python objects to be used as States, Actions and Observations.
 */

namespace thts::python {
    // forward declares
    class PyThtsEnv;
    class PyMultiprocessingThtsEnv;

    // namespace includes
    using namespace thts;
    namespace py = pybind11;

    /**
     * Python Observations
     * Wrapper around an arbitrary python object
    */
    class PyObservation : public Observation {

        protected:
            mutable std::recursive_mutex lock;
            mutable std::shared_ptr<py::object> py_obs;
            std::shared_ptr<PickleWrapper> py_pickle_wrapper;
            mutable std::shared_ptr<std::string> serialised_obs;
        
        public:
            PyObservation(std::shared_ptr<PickleWrapper> py_pickle_wrapper, std::shared_ptr<py::object> _py_obs);
            PyObservation(
                std::shared_ptr<PickleWrapper> py_pickle_wrapper, std::shared_ptr<std::string> serialised_obs);
            virtual ~PyObservation();
            bool equals(const PyObservation& other) const;

            std::shared_ptr<py::object> get_py_obs() const;
            std::shared_ptr<std::string> get_serialised_obs() const;

            virtual std::size_t hash() const override;
            virtual bool equals_itfc(const Observation& other) const override;
            virtual std::string get_pretty_print_string() const override;
    };

    /**
     * Python State
     * Wrapper around an arbitrary python object
    */
    class PyState : public State {
        friend PyThtsEnv;
        friend PyMultiprocessingThtsEnv;

        protected:
            mutable std::recursive_mutex lock;
            mutable std::shared_ptr<py::object> py_state;
            std::shared_ptr<PickleWrapper> py_pickle_wrapper;
            mutable std::shared_ptr<std::string> serialised_state;
        
        public:
            PyState(std::shared_ptr<PickleWrapper> py_pickle_wrapper, std::shared_ptr<py::object> _py_state);
            PyState(std::shared_ptr<PickleWrapper> py_pickle_wrapper, std::shared_ptr<std::string> serialised_state);
            virtual ~PyState();
            bool equals(const PyState& other) const;

            std::shared_ptr<py::object> get_py_state() const;
            std::shared_ptr<std::string> get_serialised_state() const;
            
            virtual std::size_t hash() const override;
            virtual bool equals_itfc(const Observation& other) const override;
            virtual std::string get_pretty_print_string() const override;
    };

    /**
     * Python Action
     * Wrapper around an arbitrary python object
    */
    class PyAction : public Action {
        friend PyThtsEnv;
        friend PyMultiprocessingThtsEnv;
        
        protected:
            mutable std::recursive_mutex lock;
            mutable std::shared_ptr<py::object> py_action;
            std::shared_ptr<PickleWrapper> py_pickle_wrapper;
            mutable std::shared_ptr<std::string> serialised_action;
        
        public:
            PyAction(std::shared_ptr<PickleWrapper> py_pickle_wrapper, std::shared_ptr<py::object> _py_action);
            PyAction(std::shared_ptr<PickleWrapper> py_pickle_wrapper, std::shared_ptr<std::string> serialised_action);
            virtual ~PyAction();
            bool equals(const PyAction& other) const;

            std::shared_ptr<py::object> get_py_action() const;
            std::shared_ptr<std::string> get_serialised_action() const;

            virtual std::size_t hash() const override;
            virtual bool equals_itfc(const Action& other) const override;
            virtual std::string get_pretty_print_string() const override;
    };
}

// /**
//  * Forward declaring the hash, equality and output stream functions defined in thts_types.cpp.
//  * Needed so other files know to look at thts_types.o to find implementations of these functions.
//  */
// namespace std {
//     using namespace thts::python;

//     /**
//      * Hash, equality class and output stream function definitions for PyObservation.
//      */
//     template <> 
//     struct hash<PyObservation> {
//         size_t operator()(const PyObservation&) const;
//     };

//     template <> 
//     struct hash<shared_ptr<const PyObservation>> {
//         size_t operator()(const shared_ptr<const PyObservation>&) const;
//     };
    
//     bool operator==(const PyObservation& lhs, const PyObservation& rhs);
//     bool operator==(const shared_ptr<const PyObservation>& lhs, const shared_ptr<const PyObservation>& rhs);

//     template <> 
//     struct equal_to<PyObservation> {
//         bool operator()(const PyObservation&, const PyObservation&) const;
//     };

//     template <> 
//     struct equal_to<shared_ptr<const PyObservation>> {
//         bool operator()(const shared_ptr<const PyObservation>&, const shared_ptr<const PyObservation>&) const;
//     };

//     ostream& operator<<(ostream& os, const PyObservation& observation);
//     ostream& operator<<(ostream& os, const shared_ptr<const PyObservation>& observation);

//     /**
//      * Hash, equality class and output stream function definitions for PyState.
//      */
//     template <> 
//     struct hash<PyState> {
//         size_t operator()(const PyState&) const;
//     };

//     template <> 
//     struct hash<std::shared_ptr<const PyState>> {
//         size_t operator()(const shared_ptr<const PyState>&) const;
//     };
    
//     bool operator==(const PyState& lhs, const PyState& rhs);
//     bool operator==(const shared_ptr<const PyState>& lhs, const shared_ptr<const PyState>& rhs);

//     template <> 
//     struct equal_to<PyState> {
//         bool operator()(const PyState&, const PyState&) const;
//     };

//     template <> 
//     struct equal_to<shared_ptr<const PyState>> {
//         bool operator()(const shared_ptr<const PyState>&, const shared_ptr<const PyState>&) const;
//     };

//     ostream& operator<<(ostream& os, const PyState& state);
//     ostream& operator<<(ostream& os, const shared_ptr<const PyState>& state);

//     /**
//      * Hash, equality class and output stream function definitions for PyAction.
//      */
//     template <> 
//     struct hash<PyAction> {
//         size_t operator()(const PyAction&) const;
//     };

//     template <> 
//     struct hash<shared_ptr<const PyAction>> {
//         size_t operator()(const shared_ptr<const PyAction>&) const;
//     };
    
//     bool operator==(const PyAction& lhs, const PyAction& rhs);
//     bool operator==(const shared_ptr<const PyAction>& lhs, const shared_ptr<const PyAction>& rhs);

//     template <> struct equal_to<PyAction> {
//         bool operator()(const PyAction&, const PyAction&) const;
//     };

//     template <> struct equal_to<shared_ptr<const PyAction>> {
//         bool operator()(const shared_ptr<const PyAction>&, const shared_ptr<const PyAction>&) const;
//     };

//     ostream& operator<<(ostream& os, const PyAction& action);
//     ostream& operator<<(ostream& os, const shared_ptr<const PyAction>& action);
// }