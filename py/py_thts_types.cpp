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
    PyObservation::PyObservation(shared_ptr<PickleWrapper> py_pickle_wrapper, shared_ptr<py::object> _py_obs) : 
        lock(),
        py_obs(_py_obs),
        py_pickle_wrapper(py_pickle_wrapper),
        serialised_obs()
    {
    }

    PyObservation::PyObservation(shared_ptr<PickleWrapper> py_pickle_wrapper, shared_ptr<string> serialised_obs) : 
        lock(),
        py_obs(),
        py_pickle_wrapper(py_pickle_wrapper),
        serialised_obs(serialised_obs)
    {
    }
    
    PyObservation::~PyObservation() {
        lock_guard<recursive_mutex> lg(lock);
        py_obs.reset();
        py_pickle_wrapper.reset();
    }

    shared_ptr<py::object> PyObservation::get_py_obs() const {
        if (py_obs == nullptr) {
            py_obs = make_shared<py::object>(py_pickle_wrapper->deserialise(*serialised_obs));
        }
        return py_obs;
    }

    shared_ptr<string> PyObservation::get_serialised_obs() const {
        if (serialised_obs == nullptr) {
            lock_guard<recursive_mutex> lg(lock);
            serialised_obs = make_shared<string>(py_pickle_wrapper->serialise(*py_obs));
        }
        return serialised_obs;
    }

    size_t PyObservation::hash() const {
        // lock_guard<recursive_mutex> lg(lock);
        // return py::hash(*get_py_obs());
        return std::hash<string>{}(*get_serialised_obs());
    }

    bool PyObservation::equals(const PyObservation& other) const {
        // thts::python::helper::ordered_lock(get_py_obs(), lock, other.get_py_obs(), other.lock);
        // bool result = get_py_obs()->is(*other.get_py_obs());
        // lock.unlock();
        // other.lock.unlock();
        // return result;
        return get_serialised_obs() == other.get_serialised_obs();
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
        return py::str(*get_py_obs());
    }

    /**
     * Implementation of PyState 
     * Just point towards pybind interface for each function
     */
    PyState::PyState(shared_ptr<PickleWrapper> py_pickle_wrapper, shared_ptr<py::object> _py_state) : 
        lock(),
        py_state(_py_state),
        py_pickle_wrapper(py_pickle_wrapper),
        serialised_state()
    {
    }

    PyState::PyState(shared_ptr<PickleWrapper> py_pickle_wrapper, shared_ptr<string> serialised_state) : 
        lock(),
        py_state(),
        py_pickle_wrapper(py_pickle_wrapper),
        serialised_state(serialised_state)
    {
    }
    
    PyState::~PyState() {
        lock_guard<recursive_mutex> lg(lock);
        py_state.reset();
        py_pickle_wrapper.reset();
    }

    shared_ptr<py::object> PyState::get_py_state() const {
        if (py_state == nullptr) {
            py_state = make_shared<py::object>(py_pickle_wrapper->deserialise(*serialised_state));
        }
        return py_state;
    }

    shared_ptr<string> PyState::get_serialised_state() const {
        if (serialised_state == nullptr) {
            lock_guard<recursive_mutex> lg(lock);
            serialised_state = make_shared<string>(py_pickle_wrapper->serialise(*py_state));
        }
        return serialised_state;
    }

    size_t PyState::hash() const {
        // lock_guard<recursive_mutex> lg(lock);
        // return py::hash(*get_py_state());
        return std::hash<string>{}(*get_serialised_state());
    }

    bool PyState::equals(const PyState& other) const {
        // thts::python::helper::ordered_lock(get_py_state(), lock, other.get_py_state(), other.lock);
        // bool result = get_py_state()->is(*other.get_py_state());
        // lock.unlock();
        // other.lock.unlock();
        // return result;
        return get_serialised_state() == other.get_serialised_state();
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
        return py::str(*get_py_state());
    }

    /**
     * Implementation of PyAction
     * Just point towards pybind interface for each function
     */
    PyAction::PyAction(shared_ptr<PickleWrapper> py_pickle_wrapper, shared_ptr<py::object> _py_action) : 
        lock(),
        py_action(_py_action),
        py_pickle_wrapper(py_pickle_wrapper),
        serialised_action()
    {
    }

    PyAction::PyAction(shared_ptr<PickleWrapper> py_pickle_wrapper, shared_ptr<string> serialised_action) : 
        lock(),
        py_action(),
        py_pickle_wrapper(py_pickle_wrapper),
        serialised_action(serialised_action)
    {
    }
    
    PyAction::~PyAction() {
        lock_guard<recursive_mutex> lg(lock);
        py_action.reset();
        py_pickle_wrapper.reset();
    }

    shared_ptr<py::object> PyAction::get_py_action() const {
        if (py_action == nullptr) {
            py_action = make_shared<py::object>(py_pickle_wrapper->deserialise(*serialised_action));
        }
        return py_action;
    }

    shared_ptr<string> PyAction::get_serialised_action() const {
        if (serialised_action == nullptr) {
            lock_guard<recursive_mutex> lg(lock);
            serialised_action = make_shared<string>(py_pickle_wrapper->serialise(*py_action));
        }
        return serialised_action;
    }

    size_t PyAction::hash() const {
        // lock_guard<recursive_mutex> lg(lock);
        // return py::hash(*get_py_action());
        return std::hash<string>{}(*get_serialised_action());
    }

    bool PyAction::equals(const PyAction& other) const {
        // thts::python::helper::ordered_lock(get_py_action(), lock, other.get_py_action(), other.lock);
        // bool result = get_py_action()->is(*other.get_py_action());
        // lock.unlock();
        // other.lock.unlock();
        // return result;
        return get_serialised_action() == other.get_serialised_action();
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
        return py::str(*get_py_action());
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