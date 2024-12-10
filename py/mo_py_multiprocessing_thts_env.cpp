#include "py/mo_py_multiprocessing_thts_env.h"

#include "py/py_helper.h"
#include "py/py_helper_templates.h"
#include "py/py_thts_types.h"

#include <mutex>

#include <unistd.h>

#include <iostream>

namespace py = pybind11;
using namespace std; 

/**
 * Wrapper around Python 'PyThtsEnv' object, and providing a thts 'ThtsEnv' interface for it
 * 
 * TODO: get python contexts working
 */
namespace thts::python { 

    MoPyMultiprocessingThtsEnv::MoPyMultiprocessingThtsEnv(
        shared_ptr<PickleWrapper> pickle_wrapper,
        string& thts_unique_filename,
        shared_ptr<py::object> py_thts_env,
        bool is_server_process) :
            PyMultiprocessingThtsEnv(pickle_wrapper, thts_unique_filename, py_thts_env, is_server_process),
            MoThtsEnv(
                (py_thts_env != nullptr) ? helper::call_py_getter<int>(py_thts_env,"get_reward_dim")        : 2, 
                (py_thts_env != nullptr) ? helper::call_py_getter<bool>(py_thts_env,"is_fully_observable")  : true)
    {
    }

    MoPyMultiprocessingThtsEnv::MoPyMultiprocessingThtsEnv(
        shared_ptr<PickleWrapper> pickle_wrapper,
        string& thts_unique_filename,
        string module_name,
        string class_name,
        shared_ptr<py::dict> constructor_kw_args,
        bool is_server_process) :
            PyMultiprocessingThtsEnv(pickle_wrapper, thts_unique_filename, module_name, class_name, constructor_kw_args, is_server_process),
            MoThtsEnv(2,true)
    {
        py::gil_scoped_acquire acquire;
        _is_fully_observable = py_thts_env->attr("is_fully_observable")().cast<bool>();
        reward_dim = py_thts_env->attr("get_reward_dim")().cast<int>();
    }

    MoPyMultiprocessingThtsEnv::MoPyMultiprocessingThtsEnv(MoPyMultiprocessingThtsEnv& other) :
        PyMultiprocessingThtsEnv(other),
        MoThtsEnv(other)
    {
    }

    shared_ptr<ThtsEnv> MoPyMultiprocessingThtsEnv::clone() {
        return make_shared<MoPyMultiprocessingThtsEnv>(*this);
    }

    string MoPyMultiprocessingThtsEnv::get_multiprocessing_env_type_id() 
    {
        return MOPY_ENV_SERVER_ID;
    }

    double MoPyMultiprocessingThtsEnv::get_reward_itfc(
        shared_ptr<const State> state, 
        shared_ptr<const Action> action, 
        ThtsEnvContext& ctx) const 
    {
        return MoThtsEnv::get_reward_itfc(state, action, ctx);
    }

    shared_ptr<vector<double>> MoPyMultiprocessingThtsEnv::get_reward_py_server(string& state, string& action) const 
    {
        py::object py_state = pickle_wrapper->deserialise(state);
        py::object py_action = pickle_wrapper->deserialise(action);
        py::handle py_get_reward_fn = py_thts_env->attr("get_reward");
        py::list py_reward_list = py_get_reward_fn(py_state, py_action);

        shared_ptr<vector<double>> result = make_shared<vector<double>>();
        for (py::handle py_reward_i : py_reward_list) {
            result->push_back(py_reward_i.cast<double>());
        }
        return result;
    }

    Eigen::ArrayXd MoPyMultiprocessingThtsEnv::get_mo_reward(
        shared_ptr<const PyState> state, 
        shared_ptr<const PyAction> action,
        ThtsEnvContext& ctx) const 
    {
        shared_mem_wrapper->rpc_id = RPC_get_reward;
        shared_mem_wrapper->value_type = SMT_strings;
        shared_mem_wrapper->strings = make_shared<vector<string>>();
        shared_mem_wrapper->strings->push_back(*state->get_serialised_state());
        shared_mem_wrapper->strings->push_back(*action->get_serialised_action());
        shared_mem_wrapper->make_rpc_call();

        Eigen::ArrayXd mo_reward = Eigen::ArrayXd::Zero(reward_dim);
        for (size_t i=0; i<shared_mem_wrapper->doubles->size(); i++) {
            mo_reward[i] = shared_mem_wrapper->doubles->at(i);
        }
        return mo_reward;
    }

    Eigen::ArrayXd MoPyMultiprocessingThtsEnv::get_mo_reward_itfc(
        shared_ptr<const State> state, 
        shared_ptr<const Action> action,
        ThtsEnvContext& ctx) const
    {
        ThtsEnvContext& py_ctx = (ThtsEnvContext&) ctx;
        shared_ptr<const PyState> state_itfc = static_pointer_cast<const PyState>(state);
        shared_ptr<const PyAction> action_itfc = static_pointer_cast<const PyAction>(action);
        return get_mo_reward(state_itfc, action_itfc, py_ctx); 
    }
     
    shared_ptr<ThtsEnvContext> MoPyMultiprocessingThtsEnv::sample_context_itfc(
        int tid, RandManager& rand_manager) const 
    {
        return MoThtsEnv::sample_context_itfc(tid, rand_manager);  
    }
}



/**
 * Pointing virtual functions to the correct place
*/
namespace thts::python {

    shared_ptr<const State> MoPyMultiprocessingThtsEnv::get_initial_state_itfc() const
    {
        return PyMultiprocessingThtsEnv::get_initial_state_itfc();
    }
    
    bool MoPyMultiprocessingThtsEnv::is_sink_state_itfc(shared_ptr<const State> state, ThtsEnvContext& ctx) const
    {
        return PyMultiprocessingThtsEnv::is_sink_state_itfc(state, ctx);
    }
    
    shared_ptr<ActionVector> MoPyMultiprocessingThtsEnv::get_valid_actions_itfc(
        shared_ptr<const State> state, ThtsEnvContext& ctx) const
    {
        return PyMultiprocessingThtsEnv::get_valid_actions_itfc(state, ctx);
    }
    
    shared_ptr<StateDistr> MoPyMultiprocessingThtsEnv::get_transition_distribution_itfc(
        shared_ptr<const State> state, 
        shared_ptr<const Action> action, 
        ThtsEnvContext& ctx) const
    {
        return PyMultiprocessingThtsEnv::get_transition_distribution_itfc(state, action, ctx);
    }
    
    shared_ptr<const State> MoPyMultiprocessingThtsEnv::sample_transition_distribution_itfc(
        shared_ptr<const State> state, 
        shared_ptr<const Action> action, 
            RandManager& rand_manager, 
            ThtsEnvContext& ctx) const
    {
        return PyMultiprocessingThtsEnv::sample_transition_distribution_itfc(state, action, rand_manager, ctx);
    }
    
    shared_ptr<ObservationDistr> MoPyMultiprocessingThtsEnv::get_observation_distribution_itfc(
        shared_ptr<const Action> action, 
        shared_ptr<const State> next_state, 
        ThtsEnvContext& ctx) const
    {
        return PyMultiprocessingThtsEnv::get_observation_distribution_itfc(action, next_state, ctx);
    }
    
    shared_ptr<const Observation> MoPyMultiprocessingThtsEnv::sample_observation_distribution_itfc(
        shared_ptr<const Action> action, 
        shared_ptr<const State> next_state, 
            RandManager& rand_manager, 
            ThtsEnvContext& ctx) const
    {
        return PyMultiprocessingThtsEnv::sample_observation_distribution_itfc(action, next_state, rand_manager, ctx);
    }
    
    void MoPyMultiprocessingThtsEnv::reset_itfc() const
    {
        PyMultiprocessingThtsEnv::reset_itfc();
    }
    
}