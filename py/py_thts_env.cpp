#include "py/py_thts_env.h"

#include "py/gil_helpers.h"
#include "py/py_thts_types.h"

#include <mutex>

#include <iostream>

namespace py = pybind11;
using namespace std; 

/**
 * 
 */
namespace thts::python { 
    PyThtsEnv::PyThtsEnv(py::object _py_thts_env) :
        ThtsEnv(), py_thts_env()  
    {
        thts::python::helper::GilReenterantLockGuard gil_lg;
        py_thts_env = _py_thts_env;
        _is_fully_observable = py_thts_env.attr("is_fully_observable")().cast<bool>();
    }

    shared_ptr<const PyState> PyThtsEnv::get_initial_state() const {
        lock_guard<mutex> lg(init_lock);
        thts::python::helper::GilReenterantLockGuard gil_lg;
        lock_guard<mutex> py_env_lg(py_thts_env_lock);
        py::handle py_get_initial_state_fn = py_thts_env.attr("get_initial_state");
        py::object py_init_state = py_get_initial_state_fn();
        return make_shared<const PyState>(py_init_state);
    }

    bool PyThtsEnv::is_sink_state(shared_ptr<const PyState> state) const {
        lock_guard<mutex> lg(sink_lock);
        PyState& state_non_const_ref = const_cast<PyState&>(*state);
        thts::python::helper::GilReenterantLockGuard gil_lg;
        lock_guard<mutex> py_env_lg(py_thts_env_lock);
        py::handle py_is_sink_state_fn = py_thts_env.attr("is_sink_state");
        return py_is_sink_state_fn(state_non_const_ref.py_state).cast<bool>();
    }

    shared_ptr<PyActionVector> PyThtsEnv::get_valid_actions(shared_ptr<const PyState> state) const {
        lock_guard<mutex> lg(valid_lock);
        PyState& state_non_const_ref = const_cast<PyState&>(*state);
        thts::python::helper::GilReenterantLockGuard gil_lg;
        py_thts_env_lock.lock();
        py::object py_get_valid_actions_fn = py_thts_env.attr("get_valid_actions");
        py::list py_valid_actions_list = py_get_valid_actions_fn(state_non_const_ref.py_state);
        py_thts_env_lock.unlock();
        shared_ptr<PyActionVector> action_vector = make_shared<PyActionVector>();
        for (py::handle py_action : py_valid_actions_list) {
            py::object py_action_object = py::cast<py::object>(py_action);
            action_vector->push_back(make_shared<const PyAction>(py_action_object));
        }
        return action_vector;
    }

    shared_ptr<PyStateDistr> PyThtsEnv::get_transition_distribution(
        shared_ptr<const PyState> state, shared_ptr<const PyAction> action) const 
    {
        lock_guard<mutex> lg(distr_lock);
        PyState& state_non_const_ref = const_cast<PyState&>(*state);
        PyAction& action_non_const_ref = const_cast<PyAction&>(*action);
        thts::python::helper::GilReenterantLockGuard gil_lg;
        py_thts_env_lock.lock();
        py::object py_get_transition_distribution_fn = py_thts_env.attr("get_transition_distribution");
        py::dict py_transition_prob_map = py_get_transition_distribution_fn(
            state_non_const_ref.py_state, action_non_const_ref.py_action);
        py_thts_env_lock.unlock();
        shared_ptr<PyStateDistr> transition_prob_map = make_shared<PyStateDistr>();
        for (pair<py::handle,py::handle> py_state_prob_pair : py_transition_prob_map) {
            py::object py_next_state = py::cast<py::object>(py_state_prob_pair.first);
            py::object py_prob_double = py::cast<py::object>(py_state_prob_pair.second);
            transition_prob_map->insert_or_assign(
                make_shared<const PyState>(py_next_state), py_prob_double.cast<double>());
        }
        return transition_prob_map;
    }

    shared_ptr<const PyState> PyThtsEnv::sample_transition_distribution(
        shared_ptr<const PyState> state, shared_ptr<const PyAction> action, RandManager& rand_manager) const 
    {
        lock_guard<mutex> lg(sampl_lock);
        PyState& state_non_const_ref = const_cast<PyState&>(*state);
        PyAction& action_non_const_ref = const_cast<PyAction&>(*action);
        thts::python::helper::GilReenterantLockGuard gil_lg;
        lock_guard<mutex> py_env_lg(py_thts_env_lock);
        py::handle py_sample_transition_distribution_fn = py_thts_env.attr("sample_transition_distribution");
        py::object py_next_state = py_sample_transition_distribution_fn(
            state_non_const_ref.py_state, action_non_const_ref.py_action);
        return make_shared<const PyState>(py_next_state);
    }

    double PyThtsEnv::get_reward(
        shared_ptr<const PyState> state, 
        shared_ptr<const PyAction> action, 
        shared_ptr<const PyObservation> observation) const 
    {
        lock_guard<mutex> lg(rewrd_lock);
        PyState& state_non_const_ref = const_cast<PyState&>(*state);
        PyAction& action_non_const_ref = const_cast<PyAction&>(*action);
        PyObservation& observation_non_const_ref = const_cast<PyObservation&>(*observation);
        thts::python::helper::GilReenterantLockGuard gil_lg;
        py_thts_env_lock.lock();
        py::object py_get_reward_fn = py_thts_env.attr("get_reward");
        py_thts_env_lock.unlock();
        py::handle py_state = py::cast<py::handle>(state_non_const_ref.py_state);
        py::handle py_action = py::cast<py::handle>(action_non_const_ref.py_action);
        py::handle py_obs = py::none();
        if (observation != nullptr) {
            py_obs = py::cast<py::handle>(observation_non_const_ref.py_obs);
        }
        lock_guard<mutex> py_env_lg(py_thts_env_lock);
        return py_get_reward_fn(py_state, py_action, py_obs).cast<double>();
    }

    // TODO: change this to use a pybind11 python object too for the context
    shared_ptr<PyThtsContext> PyThtsEnv::sample_context(shared_ptr<const PyState> state) const
    {
        // thts::python::helper::GilReenterantLockGuard gil_lg;
        shared_ptr<const State> state_itfc = static_pointer_cast<const State>(state);
        shared_ptr<ThtsEnvContext> context = ThtsEnv::sample_context_itfc(state_itfc);
        return static_pointer_cast<PyThtsContext>(context);
    }
}



/**
 * Boilerplate defined functions. Copied from thts_env_template.h.
 * 
 * TODO: decide if need to write a custom version of these depending on if need partial observability or if need 
 * custom contexts.
 */
namespace thts::python {
    shared_ptr<PyObservationDistr> PyThtsEnv::get_observation_distribution(
        shared_ptr<const PyAction> action, shared_ptr<const PyState> next_state) const 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<ObservationDistr> distr_itfc = ThtsEnv::get_observation_distribution_itfc(
            act_itfc, next_state_itfc);
        shared_ptr<PyObservationDistr> distr;
        for (pair<const shared_ptr<const Observation>,double> pr : *distr_itfc) {
            shared_ptr<const PyObservation> obsv = static_pointer_cast<const PyObservation>(pr.first);
            distr->insert_or_assign(obsv, pr.second);
        }
        return distr;
    }

    shared_ptr<const PyObservation> PyThtsEnv::sample_observation_distribution(
        shared_ptr<const PyAction> action, 
        shared_ptr<const PyState> next_state, 
        RandManager& rand_manager) const 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<const Observation> obsv_itfc = ThtsEnv::sample_observation_distribution_itfc(
            act_itfc, next_state_itfc, rand_manager);
        return static_pointer_cast<const PyObservation>(obsv_itfc);
    }
}



/**
 * Boilerplate ThtsEnv interface implementation. Copied from thts_env_template.h.
 * All this code basically calls the corresponding implementation function, with approprtiate casts before/after.
 */
namespace thts::python {
    
    shared_ptr<const State> PyThtsEnv::get_initial_state_itfc() const {
        shared_ptr<const PyState> init_state = get_initial_state();
        return static_pointer_cast<const State>(init_state);
    }

    bool PyThtsEnv::is_sink_state_itfc(shared_ptr<const State> state) const {
        shared_ptr<const PyState> state_itfc = static_pointer_cast<const PyState>(state);
        return is_sink_state(state_itfc);
    }

    shared_ptr<ActionVector> PyThtsEnv::get_valid_actions_itfc(shared_ptr<const State> state) const {
        shared_ptr<const PyState> state_itfc = static_pointer_cast<const PyState>(state);
        shared_ptr<vector<shared_ptr<const PyAction>>> valid_actions_itfc = get_valid_actions(state_itfc);

        shared_ptr<ActionVector> valid_actions = make_shared<ActionVector>();
        for (shared_ptr<const PyAction> act : *valid_actions_itfc) {
            valid_actions->push_back(static_pointer_cast<const Action>(act));
        }
        return valid_actions;
    }

    shared_ptr<StateDistr> PyThtsEnv::get_transition_distribution_itfc(
        shared_ptr<const State> state, shared_ptr<const Action> action) const 
    {
        shared_ptr<const PyState> state_itfc = static_pointer_cast<const PyState>(state);
        shared_ptr<const PyAction> action_itfc = static_pointer_cast<const PyAction>(action);
        shared_ptr<PyStateDistr> distr_itfc = get_transition_distribution(state_itfc, action_itfc);
        
        shared_ptr<StateDistr> distr = make_shared<StateDistr>(); 
        for (pair<shared_ptr<const PyState>,double> key_val_pair : *distr_itfc) {
            shared_ptr<const State> obsv = static_pointer_cast<const State>(key_val_pair.first);
            double prob = key_val_pair.second;
            distr->insert_or_assign(obsv, prob);
        }
        return distr;
    }

    shared_ptr<const State> PyThtsEnv::sample_transition_distribution_itfc(
       shared_ptr<const State> state, shared_ptr<const Action> action, RandManager& rand_manager) const 
    {
        shared_ptr<const PyState> state_itfc = static_pointer_cast<const PyState>(state);
        shared_ptr<const PyAction> action_itfc = static_pointer_cast<const PyAction>(action);
        shared_ptr<const PyState> obsv = sample_transition_distribution(state_itfc, action_itfc, rand_manager);
        return static_pointer_cast<const State>(obsv);
    }

    shared_ptr<ObservationDistr> PyThtsEnv::get_observation_distribution_itfc(
        shared_ptr<const Action> action, shared_ptr<const State> next_state) const
    {
        shared_ptr<const PyAction> act_itfc = static_pointer_cast<const PyAction>(action);
        shared_ptr<const PyState> next_state_itfc = static_pointer_cast<const PyState>(next_state);
        shared_ptr<PyObservationDistr> distr_itfc = get_observation_distribution(
            act_itfc, next_state_itfc);
        shared_ptr<ObservationDistr> distr;
        for (pair<const shared_ptr<const PyObservation>,double> pr : *distr_itfc) {
            shared_ptr<const Observation> obsv = static_pointer_cast<const Observation>(pr.first);
            distr->insert_or_assign(obsv, pr.second);
        }
        return distr;
    }     

    shared_ptr<const Observation> PyThtsEnv::sample_observation_distribution_itfc(
        shared_ptr<const Action> action, 
        shared_ptr<const State> next_state,
         RandManager& rand_manager) const
    {
        shared_ptr<const PyAction> act_itfc = static_pointer_cast<const PyAction>(action);
        shared_ptr<const PyState> next_state_itfc = static_pointer_cast<const PyState>(next_state);
        shared_ptr<const PyObservation> obsv_itfc = sample_observation_distribution(
            act_itfc, next_state_itfc, rand_manager);
        return static_pointer_cast<const Observation>(obsv_itfc);
    }

    double PyThtsEnv::get_reward_itfc(
        shared_ptr<const State> state, 
        shared_ptr<const Action> action, 
        shared_ptr<const Observation> observation) const
    {
        shared_ptr<const PyState> state_itfc = static_pointer_cast<const PyState>(state);
        shared_ptr<const PyAction> action_itfc = static_pointer_cast<const PyAction>(action);
        shared_ptr<const PyObservation> obsv_itfc = static_pointer_cast<const PyObservation>(observation);
        return get_reward(state_itfc, action_itfc, obsv_itfc); 
    }

    shared_ptr<ThtsEnvContext> PyThtsEnv::sample_context_itfc(shared_ptr<const State> state) const
    {
        shared_ptr<const PyState> state_itfc = static_pointer_cast<const PyState>(state);
        shared_ptr<PyThtsContext> context = sample_context(state_itfc);
        return static_pointer_cast<ThtsEnvContext>(context);
    }
}