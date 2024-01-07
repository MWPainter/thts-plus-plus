#include "py/py_thts_types.h"

#include "py/py_helper_templates.h"

#include <functional> 
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;
using namespace std;
using namespace thts;
using namespace thts::python;




namespace thts::python { 
    /**
     * Implementation of PyObservation 
     * Just point towards pybind interface for each function
     */
    PyObservation::PyObservation(shared_ptr<py::object> _py_obs) : py_obs(), lock()
    {
        lock_guard<recursive_mutex> lg(lock);
        py_obs = _py_obs;
    }

    PyObservation::~PyObservation() {
        lock_guard<recursive_mutex> lg(lock);
        py_obs.reset();
    }

    size_t PyObservation::hash() const {
        lock_guard<recursive_mutex> lg(lock);
        return py::hash(*py_obs);
    }
    
    bool PyObservation::equals(const PyObservation& other) const {
        thts::python::helper::ordered_lock(py_obs, lock, other.py_obs, other.lock);
        bool result = py_obs->is(*other.py_obs);
        lock.unlock();
        other.lock.unlock();
        return result;
    }

    bool PyObservation::equals_itfc(const Observation& other) const {
        try {
            const PyObservation& oth = dynamic_cast<const PyObservation&>(other);
            return equals(oth);
        }
        catch (const bad_cast&) {
            return false;
        }
    }
    
    string PyObservation::get_pretty_print_string() const {
        lock_guard<recursive_mutex> lg(lock);
        return py::str(*py_obs);
    }

    /**
     * Implementation of PyState 
     * Just point towards pybind interface for each function
     */
    PyState::PyState(shared_ptr<py::object> _py_state) : py_state(), lock()
    {
        lock_guard<recursive_mutex> lg(lock);
        py_state = _py_state;
    }

    PyState::~PyState() {
        lock_guard<recursive_mutex> lg(lock);
        py_state.reset();
    }

    size_t PyState::hash() const {
        lock_guard<recursive_mutex> lg(lock);
        return py::hash(*py_state);
    } 
    
    bool PyState::equals(const PyState& other) const {
        thts::python::helper::ordered_lock(py_state, lock, other.py_state, other.lock);
        bool result = py_state->is(*other.py_state);
        lock.unlock();
        other.lock.unlock();
        return result;
    }

    bool PyState::equals_itfc(const Observation& other) const {
        try {
            const PyState& oth = dynamic_cast<const PyState&>(other);
            return equals(oth);
        }
        catch (const bad_cast&) {
            return false;
        }
    }
    
    string PyState::get_pretty_print_string() const {
        lock_guard<recursive_mutex> lg(lock);
        return py::str(*py_state);
    }

    /**
     * Implementation of PyAction
     * Just point towards pybind interface for each function
     */
    PyAction::PyAction(shared_ptr<py::object> _py_action) : py_action(), lock()
    {
        lock_guard<recursive_mutex> lg(lock);
        py_action = _py_action;
    }

    PyAction::~PyAction() {
        lock_guard<recursive_mutex> lg(lock);
        py_action.reset();
    }
    
    size_t PyAction::hash() const {
        lock_guard<recursive_mutex> lg(lock);
        return py::hash(*py_action);
    }
    
    bool PyAction::equals(const PyAction& other) const {
        thts::python::helper::ordered_lock(py_action, lock, other.py_action, other.lock);
        bool result = py_action->is(*other.py_action);
        lock.unlock();
        other.lock.unlock();
        return result;
    }

    bool PyAction::equals_itfc(const Action& other) const {
        try {
            const PyAction& oth = dynamic_cast<const PyAction&>(other);
            return equals(oth);
        }
        catch (const bad_cast&) {
            return false;
        }
    }
    
    string PyAction::get_pretty_print_string() const {
        lock_guard<recursive_mutex> lg(lock);
        return py::str(*py_action);
    }
}

/**
 * Implementation of hash, equal_to and << operator functions for PyState/PyAction/PyObservation. 
 */
namespace std {
    /**
     * Implementation of std::hash<PyObservation>, calling the virtual hash function.
     */
    size_t hash<PyObservation>::operator()(const PyObservation& observation) const {
        return observation.hash();
    }

    size_t hash<shared_ptr<const PyObservation>>::operator()(const shared_ptr<const PyObservation>& observation) const {
        return observation->hash();
    }

    /**
     * Implementation of std::equal_to<PyObservation>, calling the equals function.
     */
    bool operator==(const PyObservation& lhs, const PyObservation& rhs) {
        return lhs.equals(rhs);
    }

    bool operator==(const shared_ptr<const PyObservation>& lhs, const shared_ptr<const PyObservation>& rhs) {
        return lhs->equals(*rhs);
    }

    bool equal_to<PyObservation>::operator()(const PyObservation& lhs, const PyObservation& rhs) const {
        return lhs.equals(rhs);
    }

    bool equal_to<shared_ptr<const PyObservation>>::operator()(
        const shared_ptr<const PyObservation>& lhs, const shared_ptr<const PyObservation>& rhs) const 
    {
        return lhs->equals(*rhs);
    }

    /**
     * Override output stream << operator for PyObservation, using get_pretty_print_string function.
     */
    ostream& operator<<(ostream& os, const PyObservation& observation) {
        os << observation.get_pretty_print_string();
        return os;
    }

    ostream& operator<<(ostream& os, const shared_ptr<const PyObservation>& observation) {
        os << observation->get_pretty_print_string();
        return os;
    }

    /**
     * Implementation of std::hash<PyState>, calling the virtual hash function.
     */
    size_t hash<PyState>::operator()(const PyState& state) const {
        return state.hash();
    }
    
    size_t hash<shared_ptr<const PyState>>::operator()(const shared_ptr<const PyState>& state) const {
        return state->hash();
    }
    /**
     * Implementation of std::equal_to<PyState>, calling the equals function.
     */
    bool operator==(const PyState& lhs, const PyState& rhs) {
        return lhs.equals(rhs);
    }

    bool operator==(const shared_ptr<const PyState>& lhs, const shared_ptr<const PyState>& rhs) {
        return lhs->equals(*rhs);
    }

    bool equal_to<PyState>::operator()(const PyState& lhs, const PyState& rhs) const {
        return lhs.equals(rhs);
    }

    bool equal_to<shared_ptr<const PyState>>::operator()(
        const shared_ptr<const PyState>& lhs, const shared_ptr<const PyState>& rhs) const 
    {
        return lhs->equals(*rhs);
    }

    /**
     * Override output stream << operator for PyState, using get_pretty_print_string function.
     */
    ostream& operator<<(ostream& os, const PyState& state) {
        os << state.get_pretty_print_string();
        return os;
    }

    ostream& operator<<(ostream& os, const shared_ptr<const PyState>& state) {
        os << state->get_pretty_print_string();
        return os;
    }

    /**
     * Implementation of std::hash<PyAction>, calling the virtual hash function.
     */
    size_t hash<PyAction>::operator()(const PyAction& action) const {
        return action.hash();
    }

    size_t hash<shared_ptr<const PyAction>>::operator()(const shared_ptr<const PyAction>& action) const {
        return action->hash();
    }

    /**
     * Implementation of std::equal_to<PyAction>, calling the equals function.
     */
    bool operator==(const PyAction& lhs, const PyAction& rhs) {
        return lhs.equals(rhs);
    }

    bool operator==(const shared_ptr<const PyAction>& lhs, const shared_ptr<const PyAction>& rhs) {
        return lhs->equals(*rhs);
    }

    bool equal_to<PyAction>::operator()(const PyAction& lhs, const PyAction& rhs) const {
        return lhs.equals(rhs);
    }

    bool equal_to<shared_ptr<const PyAction>>::operator()(
        const shared_ptr<const PyAction>& lhs, const shared_ptr<const PyAction>& rhs) const 
    {
        return lhs->equals(*rhs);
    }

    /**
     * Override output stream << operator for PyAction, using get_pretty_print_string function.
     */
    ostream& operator<<(ostream& os, const PyAction& action) {
        os << action.get_pretty_print_string();
        return os;
    }

    ostream& operator<<(ostream& os, const shared_ptr<const PyAction>& action) {
        os << action->get_pretty_print_string();
        return os;
    }
}