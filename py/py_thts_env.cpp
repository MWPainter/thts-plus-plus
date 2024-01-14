#include "py/py_thts_env.h"

#include "py/py_helper.h"
#include "py/py_thts_context.h"
#include "py/py_thts_types.h"

#include <mutex>

#include <iostream>

namespace py = pybind11;
using namespace std; 

/**
 * Wrapper around Python 'PyThtsEnv' object, and providing a thts 'ThtsEnv' interface for it
 * 
 * TODO: get python contexts working
 */
namespace thts::python { 
    PyThtsEnv::PyThtsEnv(
        shared_ptr<py::object> py_thts_env, 
        bool multiple_threads_using_this_env,
        shared_ptr<PickleWrapper> pickle_wrapper) :
            ThtsEnv(true), 
            multiple_threads_using_this_env(multiple_threads_using_this_env), 
            py_thts_env_lock(), 
            py_thts_env(py_thts_env),
            pickle_wrapper(pickle_wrapper) 
    {
        unique_lock<mutex> lg = maybe_lock_for_py_thts_env();
        _is_fully_observable = py_thts_env->attr("is_fully_observable")().cast<bool>();
    }

    PyThtsEnv::PyThtsEnv(PyThtsEnv& other) :
        ThtsEnv(other._is_fully_observable),
        multiple_threads_using_this_env(other.multiple_threads_using_this_env),
        py_thts_env_lock(),
        py_thts_env(),
        pickle_wrapper(other.pickle_wrapper)
    {
        unique_lock<mutex> this_py_env_lg = maybe_lock_for_py_thts_env();
        unique_lock<mutex> other_py_env_lg = other.maybe_lock_for_py_thts_env();
        py::object py_thts_env_object = other.py_thts_env->attr("clone")();
        py_thts_env = make_shared<py::object>(py_thts_env_object);
    }

    shared_ptr<ThtsEnv> PyThtsEnv::clone() {
        return make_shared<PyThtsEnv>(*this);
    }

    PyThtsEnv::~PyThtsEnv() {
        lock_guard<mutex> lg(py_thts_env_lock);
        py_thts_env.reset();
    }

    unique_lock<mutex> PyThtsEnv::maybe_lock_for_py_thts_env() const {
        if (multiple_threads_using_this_env) {
            return unique_lock<mutex>(py_thts_env_lock);
        }
        return unique_lock<mutex>();
    }

    void PyThtsEnv::ensure_py_thts_env_unlocked(std::unique_lock<std::mutex>& ul) const {
        if (multiple_threads_using_this_env) {
            ul.unlock();
        }
    }

    shared_ptr<const PyState> PyThtsEnv::get_initial_state() const {
        unique_lock<mutex> py_env_lg = maybe_lock_for_py_thts_env();
        py::handle py_get_initial_state_fn = py_thts_env->attr("get_initial_state");
        py::object py_init_state = py_get_initial_state_fn();
        return make_shared<const PyState>(pickle_wrapper, make_shared<py::object>(py_init_state));
    }

    bool PyThtsEnv::is_sink_state(shared_ptr<const PyState> state, PyThtsContext& ctx) const {
        PyState& state_non_const_ref = const_cast<PyState&>(*state);
        lock_guard<recursive_mutex> state_lg(state->lock);

        // py::handle py_ctx = *ctx.py_context;
        py::handle py_state = *state_non_const_ref.py_state;

        unique_lock<mutex> py_env_lg = maybe_lock_for_py_thts_env();
        py::handle py_is_sink_state_fn = py_thts_env->attr("is_sink_state");
        // return py_is_sink_state_fn(py_state, py_ctx).cast<bool>();
        return py_is_sink_state_fn(py_state).cast<bool>();
    }

    shared_ptr<PyActionVector> PyThtsEnv::get_valid_actions(
        shared_ptr<const PyState> state, PyThtsContext& ctx) const 
    {
        PyState& state_non_const_ref = const_cast<PyState&>(*state);
        lock_guard<recursive_mutex> state_lg(state_non_const_ref.lock);

        // py::handle py_ctx = *ctx.py_context;
        py::handle py_state = *state_non_const_ref.py_state;

        unique_lock<mutex> py_env_lg = maybe_lock_for_py_thts_env();
        py::handle py_get_valid_actions_fn = py_thts_env->attr("get_valid_actions");
        // py::list py_valid_actions_list = py_get_valid_actions_fn(*state_non_const_ref.py_state, py_ctx);
        py::list py_valid_actions_list = py_get_valid_actions_fn(py_state);
        ensure_py_thts_env_unlocked(py_env_lg);

        shared_ptr<PyActionVector> action_vector = make_shared<PyActionVector>();
        for (py::handle py_action : py_valid_actions_list) {
            py::object py_action_object = py::cast<py::object>(py_action);
            action_vector->push_back(
                make_shared<const PyAction>(pickle_wrapper, make_shared<py::object>(py_action_object)));
        }
        return action_vector; 
    }

    shared_ptr<PyStateDistr> PyThtsEnv::get_transition_distribution(
        shared_ptr<const PyState> state, shared_ptr<const PyAction> action, PyThtsContext& ctx) const 
    {
        PyState& state_non_const_ref = const_cast<PyState&>(*state);
        PyAction& action_non_const_ref = const_cast<PyAction&>(*action);
        lock_guard<recursive_mutex> state_lg(state_non_const_ref.lock);
        lock_guard<recursive_mutex> action_lg(action_non_const_ref.lock);

        // py::handle py_ctx = *ctx.py_context;
        py::handle py_state = *state_non_const_ref.py_state;
        py::handle py_action =*action_non_const_ref.py_action;

        unique_lock<mutex> py_env_lg = maybe_lock_for_py_thts_env();
        py::handle py_get_transition_distribution_fn = py_thts_env->attr("get_transition_distribution");
        // py::dict py_transition_prob_map = py_get_transition_distribution_fn(py_state, py_action, py_ctx);
        py::dict py_transition_prob_map = py_get_transition_distribution_fn(py_state, py_action);
        ensure_py_thts_env_unlocked(py_env_lg);

        shared_ptr<PyStateDistr> transition_prob_map = make_shared<PyStateDistr>();
        for (pair<py::handle,py::handle> py_state_prob_pair : py_transition_prob_map) {
            py::object py_next_state = py::cast<py::object>(py_state_prob_pair.first);
            py::handle py_prob_double = py_state_prob_pair.second;
            transition_prob_map->insert_or_assign(
                make_shared<const PyState>(pickle_wrapper, make_shared<py::object>(py_next_state)), 
                py_prob_double.cast<double>());
        }
        return transition_prob_map;
    }

    shared_ptr<const PyState> PyThtsEnv::sample_transition_distribution(
        shared_ptr<const PyState> state, 
        shared_ptr<const PyAction> action, 
        RandManager& rand_manager, 
        PyThtsContext& ctx) const 
    {
        PyState& state_non_const_ref = const_cast<PyState&>(*state);
        PyAction& action_non_const_ref = const_cast<PyAction&>(*action);
        lock_guard<recursive_mutex> state_lg(state_non_const_ref.lock);
        lock_guard<recursive_mutex> action_lg(action_non_const_ref.lock);

        // py::handle py_ctx = *ctx.py_context;
        py::handle py_state = *state_non_const_ref.py_state;
        py::handle py_action =*action_non_const_ref.py_action;
        
        unique_lock<mutex> py_env_lg = maybe_lock_for_py_thts_env();
        py::handle py_sample_transition_distribution_fn = py_thts_env->attr("sample_transition_distribution");
        // py::object py_next_state = py_sample_transition_distribution_fn(py_state, py_action, py_ctx);
        py::object py_next_state = py_sample_transition_distribution_fn(py_state, py_action);
        ensure_py_thts_env_unlocked(py_env_lg);

        return make_shared<const PyState>(pickle_wrapper, make_shared<py::object>(py_next_state));
    }

    double PyThtsEnv::get_reward(
        shared_ptr<const PyState> state, 
        shared_ptr<const PyAction> action, 
        PyThtsContext& ctx) const 
    {
        PyState& state_non_const_ref = const_cast<PyState&>(*state);
        PyAction& action_non_const_ref = const_cast<PyAction&>(*action);
        lock_guard<recursive_mutex> state_lg(state_non_const_ref.lock);
        lock_guard<recursive_mutex> action_lg(action_non_const_ref.lock);

        // py::handle py_ctx = *ctx.py_context;
        py::handle py_state = *state_non_const_ref.py_state;
        py::handle py_action =*action_non_const_ref.py_action;
        
        unique_lock<mutex> py_env_lg = maybe_lock_for_py_thts_env();
        py::handle py_get_reward_fn = py_thts_env->attr("get_reward");
        // return py_get_reward_fn(py_state, py_action, py_ctx).cast<double>();
        return py_get_reward_fn(py_state, py_action).cast<double>();
    }

    shared_ptr<PyThtsContext> PyThtsEnv::sample_context_and_reset(int tid) const
    {
        unique_lock<mutex> py_env_lg = maybe_lock_for_py_thts_env();
        py::handle py_sample_context_and_reset_fn = py_thts_env->attr("sample_context_and_reset");
        py::object py_context = py_sample_context_and_reset_fn(tid);
        ensure_py_thts_env_unlocked(py_env_lg); 

        return make_shared<PyThtsContext>(make_shared<py::object>(py_context));
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
        shared_ptr<const PyAction> action, shared_ptr<const PyState> next_state, PyThtsContext& ctx) const 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<ObservationDistr> distr_itfc = ThtsEnv::get_observation_distribution_itfc(
            act_itfc, next_state_itfc, ctx);
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
        RandManager& rand_manager, 
        PyThtsContext& ctx) const 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<const Observation> obsv_itfc = ThtsEnv::sample_observation_distribution_itfc(
            act_itfc, next_state_itfc, rand_manager, ctx);
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

    bool PyThtsEnv::is_sink_state_itfc(shared_ptr<const State> state, ThtsEnvContext& ctx) const {
        PyThtsContext& py_ctx = (PyThtsContext&) ctx;
        shared_ptr<const PyState> state_itfc = static_pointer_cast<const PyState>(state);
        return is_sink_state(state_itfc, py_ctx);
    }

    shared_ptr<ActionVector> PyThtsEnv::get_valid_actions_itfc(shared_ptr<const State> state, ThtsEnvContext& ctx) const {
        PyThtsContext& py_ctx = (PyThtsContext&) ctx;
        shared_ptr<const PyState> state_itfc = static_pointer_cast<const PyState>(state);
        shared_ptr<vector<shared_ptr<const PyAction>>> valid_actions_itfc = get_valid_actions(state_itfc, py_ctx);

        shared_ptr<ActionVector> valid_actions = make_shared<ActionVector>();
        for (shared_ptr<const PyAction> act : *valid_actions_itfc) {
            valid_actions->push_back(static_pointer_cast<const Action>(act));
        }
        return valid_actions;
    }

    shared_ptr<StateDistr> PyThtsEnv::get_transition_distribution_itfc(
        shared_ptr<const State> state, shared_ptr<const Action> action, ThtsEnvContext& ctx) const 
    {
        PyThtsContext& py_ctx = (PyThtsContext&) ctx;
        shared_ptr<const PyState> state_itfc = static_pointer_cast<const PyState>(state);
        shared_ptr<const PyAction> action_itfc = static_pointer_cast<const PyAction>(action);
        shared_ptr<PyStateDistr> distr_itfc = get_transition_distribution(state_itfc, action_itfc, py_ctx);
        
        shared_ptr<StateDistr> distr = make_shared<StateDistr>(); 
        for (pair<shared_ptr<const PyState>,double> key_val_pair : *distr_itfc) {
            shared_ptr<const State> obsv = static_pointer_cast<const State>(key_val_pair.first);
            double prob = key_val_pair.second;
            distr->insert_or_assign(obsv, prob);
        }
        return distr;
    }

    shared_ptr<const State> PyThtsEnv::sample_transition_distribution_itfc(
       shared_ptr<const State> state, 
       shared_ptr<const Action> action, 
       RandManager& rand_manager, 
       ThtsEnvContext& ctx) const 
    {
        PyThtsContext& py_ctx = (PyThtsContext&) ctx;
        shared_ptr<const PyState> state_itfc = static_pointer_cast<const PyState>(state);
        shared_ptr<const PyAction> action_itfc = static_pointer_cast<const PyAction>(action);
        shared_ptr<const PyState> obsv = sample_transition_distribution(state_itfc, action_itfc, rand_manager, py_ctx);
        return static_pointer_cast<const State>(obsv);
    }

    shared_ptr<ObservationDistr> PyThtsEnv::get_observation_distribution_itfc(
        shared_ptr<const Action> action, shared_ptr<const State> next_state, ThtsEnvContext& ctx) const
    {
        PyThtsContext& py_ctx = (PyThtsContext&) ctx;
        shared_ptr<const PyAction> act_itfc = static_pointer_cast<const PyAction>(action);
        shared_ptr<const PyState> next_state_itfc = static_pointer_cast<const PyState>(next_state);
        shared_ptr<PyObservationDistr> distr_itfc = get_observation_distribution(
            act_itfc, next_state_itfc, py_ctx);
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
         RandManager& rand_manager, 
         ThtsEnvContext& ctx) const
    {
        PyThtsContext& py_ctx = (PyThtsContext&) ctx;
        shared_ptr<const PyAction> act_itfc = static_pointer_cast<const PyAction>(action);
        shared_ptr<const PyState> next_state_itfc = static_pointer_cast<const PyState>(next_state);
        shared_ptr<const PyObservation> obsv_itfc = sample_observation_distribution(
            act_itfc, next_state_itfc, rand_manager, py_ctx);
        return static_pointer_cast<const Observation>(obsv_itfc);
    }

    double PyThtsEnv::get_reward_itfc(
        shared_ptr<const State> state, 
        shared_ptr<const Action> action, 
        ThtsEnvContext& ctx) const
    {
        PyThtsContext& py_ctx = (PyThtsContext&) ctx;
        shared_ptr<const PyState> state_itfc = static_pointer_cast<const PyState>(state);
        shared_ptr<const PyAction> action_itfc = static_pointer_cast<const PyAction>(action);
        return get_reward(state_itfc, action_itfc, py_ctx); 
    }

    shared_ptr<ThtsEnvContext> PyThtsEnv::sample_context_and_reset_itfc(int tid) const
    {
        shared_ptr<PyThtsContext> context = sample_context_and_reset(tid);
        return static_pointer_cast<ThtsEnvContext>(context);
    }
}