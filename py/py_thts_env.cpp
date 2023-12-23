#include "py_thts_env.h"

namespace py = pybind11;
using namespace std; 

/**
 * TODO: implement your class here.
 */
namespace thts::py {
    PyThtsEnv::PyThtsEnv(py::object py_thts_env) :
        ThtsEnv(py_thts_env.attr("is_fully_observable")()), py_thts_env(py_thts_env) 
    {
    }

    shared_ptr<const PyState> PyThtsEnv::get_initial_state() const {
        py::object get_initial_state_fn = py_thts_env.attr("get_initial_state");
        py::object init_state = get_initial_state_fn();
        return make_shared<const PyState>(init_state);
    }

    bool PyThtsEnv::is_sink_state(shared_ptr<const PyState> state) const {
        py::object is_sink_state_fn = py_thts_env.attr("is_sink_state");
        return py::cast<bool>(is_sink_state_fn(state.py_state));
    }

    shared_ptr<PyActionVector> PyThtsEnv::get_valid_actions(shared_ptr<const PyState> state) const {
        py::object get_valid_actions_fn = py_thts_env.attr("get_valid_actions");
        py::list valid_actions_list = get_valid_actions_fn(state.py_state);
        shared_ptr<PyActionVector> action_vector = make_shared<PyActionVector>();
        for (py::object action : valid_actions_list) {
            action_vector->push_back(make_shared<const PyAction>(action));
        }
        return action_vector;
    }

    shared_ptr<PyStateDistr> PyThtsEnv::get_transition_distribution(
        shared_ptr<const PyState> state, shared_ptr<const PyAction> action) const 
    {
        py::object get_transition_distribution_fn = py_thts_env.attr("get_transition_distribution");
        py::dict py_transition_prob_map = get_valid_actions_fn(state.py_state, action.py_action);
        shared_ptr<PyStateDistr> transition_prob_map = make_shared<PyStateDistr>();
        for (auto state_prob_pair : py_transition_prob_map) {
            transition_prob_map->insert_or_assign(state_prob_pair.first, state_prob_pair.second);
        }
        return transition_prob_map;
    }

    shared_ptr<const PyState> PyThtsEnv::sample_transition_distribution(
        shared_ptr<const PyState> state, shared_ptr<const PyAction> action, RandManager& rand_manager) const 
    {
        py::object sample_transition_distribution_fn = py_thts_env.attr("sample_transition_distribution");
        py::object next_state = sample_transition_distribution_fn(state.py_state, action.py_action);
        return make_shared<const PyState>(next_state);
    }

    double PyThtsEnv::get_reward(
        shared_ptr<const PyState> state, 
        shared_ptr<const PyAction> action, 
        shared_ptr<const PyObservation> observation) const 
    {
        py::object get_reward_fn = py_thts_env.attr("get_reward");
        return get_reward_fn(state.py_state, action.py_action, observation.py_obs);
    }
}



/**
 * Boilerplate defined functions. Copied from thts_env_template.h.
 * 
 * TODO: decide if need to write a custom version of these depending on if need partial observability or if need 
 * custom contexts.
 */
namespace thts::py {
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

    shared_ptr<PyThtsContext> PyThtsEnv::sample_context(shared_ptr<const PyState> state) const
    {
        shared_ptr<const State> state_itfc = static_pointer_cast<const State>(state);
        shared_ptr<ThtsEnvContext> context = ThtsEnv::sample_context_itfc(state_itfc);
        return static_pointer_cast<PyThtsContext>(context);
    }
}



/**
 * Boilerplate ThtsEnv interface implementation. Copied from thts_env_template.h.
 * All this code basically calls the corresponding implementation function, with approprtiate casts before/after.
 */
namespace thts::py {
    
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