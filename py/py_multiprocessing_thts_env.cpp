#include "py/py_multiprocessing_thts_env.h"

#include "py/py_helper.h"
#include "py/py_helper_templates.h"
#include "py/py_thts_types.h"

#include <filesystem>
#include <mutex>

#include <signal.h>
#include <sys/prctl.h>
#include <unistd.h>

#include <iostream>
#include "helper_templates.h"

namespace py = pybind11;
using namespace std; 

/**
 * Wrapper around Python 'PyThtsEnv' object, and providing a thts 'ThtsEnv' interface for it
 * 
 * TODO: get python contexts working
 */
namespace thts::python { 

    PyMultiprocessingThtsEnv::PyMultiprocessingThtsEnv(
        shared_ptr<PickleWrapper> pickle_wrapper,
        shared_ptr<py::object> py_thts_env,
        bool is_server_process) :
            ThtsEnv((py_thts_env != nullptr) ? helper::call_py_getter<bool>(py_thts_env,"is_fully_observable") : true),
            py_thts_env(py_thts_env),
            pickle_wrapper(pickle_wrapper),
            shared_mem_wrapper(),
            module_name(),
            class_name(),
            constructor_kw_args(nullptr),
            is_server_process(is_server_process)
    {
    }

    PyMultiprocessingThtsEnv::PyMultiprocessingThtsEnv(
        shared_ptr<PickleWrapper> pickle_wrapper,
        string module_name,
        string class_name,
        shared_ptr<py::dict> constructor_kw_args,
        bool is_server_process) :
            ThtsEnv(),
            py_thts_env(nullptr),
            pickle_wrapper(pickle_wrapper),
            shared_mem_wrapper(),
            module_name(module_name),
            class_name(class_name),
            constructor_kw_args(constructor_kw_args),
            is_server_process(is_server_process)
    {
        py::gil_scoped_acquire acquire;
        py::module_ py_thts_env_module = py::module_::import(module_name.c_str());
        py::dict& kw_args = *constructor_kw_args;
        py_thts_env = make_shared<py::object>(py_thts_env_module.attr(class_name.c_str())(**kw_args));
        _is_fully_observable = py_thts_env->attr("is_fully_observable")().cast<bool>();
    }

    PyMultiprocessingThtsEnv::PyMultiprocessingThtsEnv(PyMultiprocessingThtsEnv& other) :
        ThtsEnv(other._is_fully_observable),
        py_thts_env(other.py_thts_env),
        pickle_wrapper(other.pickle_wrapper),
        shared_mem_wrapper(),
        module_name(other.module_name),
        class_name(other.class_name),
        constructor_kw_args(other.constructor_kw_args),
        is_server_process(other.is_server_process)
    {
    }

    shared_ptr<ThtsEnv> PyMultiprocessingThtsEnv::clone() {
        return make_shared<PyMultiprocessingThtsEnv>(*this);
    }

    void PyMultiprocessingThtsEnv::clear_unix_sem_and_shm() {
        shared_mem_wrapper.reset();
    }

    PyMultiprocessingThtsEnv::~PyMultiprocessingThtsEnv() {
        if (!is_server_process) {
            shared_mem_wrapper->rpc_id = RPC_kill_server;
            shared_mem_wrapper->num_args = 0;
            shared_mem_wrapper->make_kill_rpc_call();
        }

        py::gil_scoped_acquire acquire;
        py_thts_env.reset();
        pickle_wrapper.reset();
        shared_mem_wrapper.reset();
        constructor_kw_args.reset();
    }

    /**
     * Setup python server fork
     * Using: https://stackoverflow.com/questions/284325/how-to-make-child-process-die-after-parent-exits/36945270#36945270
     * -> this sets the forked process to be terminated when parent process is terminated
     * -> 
     */
    void PyMultiprocessingThtsEnv::start_python_server(int tid) {
        shared_mem_wrapper = make_shared<SharedMemWrapper>(tid, 8*1024);
        pid_t ppid_before_fork = getpid();
        pid_t pid = fork();
        if (pid == 0) {
            int r = prctl(PR_SET_PDEATHSIG, SIGTERM); 
            if (r == -1 || getppid() != ppid_before_fork) { 
                exit(-1); 
            }

            // Execve the server process if we are child process
            string py_env_server_program_name = filesystem::current_path().string() + "/py_env_server";
            vector<string> args;
            fill_multiprocessing_args(args, tid);
            vector<char*> c_args;
            c_args.push_back(const_cast<char*>(py_env_server_program_name.c_str()));
            // for (string arg : args) {  - urgh, arg is temporary storage so arg.c_str get clobbered if do this...
            for (size_t i=0; i<args.size(); i++) {
                c_args.push_back(const_cast<char*>(args[i].c_str()));
            }
            c_args.push_back(nullptr);
            execvp(c_args[0], c_args.data());
            throw runtime_error("Error in calling execvp to spawn server process.");
        }
    }

    /**
     * Gets the id to identify what type of python multiprocessing env this is
     */
    string PyMultiprocessingThtsEnv::get_multiprocessing_env_type_id() 
    {
        return PY_ENV_SERVER_ID;
    }

    /**
     * Adds the arguments needed in to run the "py_env_server" program for this env.
     */
    void PyMultiprocessingThtsEnv::fill_multiprocessing_args(vector<string>& args, int tid)
    {
        if (constructor_kw_args == nullptr) {
            throw runtime_error("Trying to start multiprocessing server without having args to construct the env "
                "in the new process.");
        }
        args.push_back(get_multiprocessing_env_type_id());
        args.push_back(to_string(tid));
        args.push_back(module_name);
        args.push_back(class_name);
        py::gil_scoped_acquire acquire;
        for (auto kw_arg : *constructor_kw_args) {
            args.push_back(kw_arg.first.cast<string>());
            args.push_back(kw_arg.second.cast<string>());
        }
    }

    /**
     * Runs the server in a loop
     * As the server runs in a forked process, so contains a completely duplicated python interpreter, we can afford to 
     * hold the gil in the forked python interpreter for the duration of the server process
     */
    void PyMultiprocessingThtsEnv::server_main(int tid) {
        shared_mem_wrapper = make_shared<SharedMemWrapper>(tid, 8*1024, true);
        while(true) {
            shared_mem_wrapper->server_wait_for_rpc_call();
            int rpc_id = shared_mem_wrapper->rpc_id;

            if (rpc_id == RPC_kill_server) {
                return;
            } else if (rpc_id == RPC_get_initial_state) {
                shared_mem_wrapper->args[0] = get_initial_state_py_server();
            } else if (rpc_id == RPC_is_sink_state) {
                string& state = shared_mem_wrapper->args[0];
                shared_mem_wrapper->args[0] = is_sink_state_py_server(state);
            } else if (rpc_id == RPC_get_valid_actions) {
                string& state = shared_mem_wrapper->args[0];
                shared_mem_wrapper->args[0] = get_valid_actions_py_server(state);
            } else if (rpc_id == RPC_get_transition_distribution) {
                string& state = shared_mem_wrapper->args[0];
                string& action = shared_mem_wrapper->args[1];
                shared_mem_wrapper->args[0] = get_transition_distribution_py_server(state, action);
            } else if (rpc_id == RPC_sample_transition_distribution) {
                string& state = shared_mem_wrapper->args[0];
                string& action = shared_mem_wrapper->args[1];
                shared_mem_wrapper->args[0] = sample_transition_distribution_py_server(state, action);
            } else if (rpc_id == RPC_get_reward) {
                string& state = shared_mem_wrapper->args[0];
                string& action = shared_mem_wrapper->args[1];
                shared_mem_wrapper->args[0] = get_reward_py_server(state, action);
            } else if (rpc_id == RPC_reset) {
                reset_py_server();
                shared_mem_wrapper->args[0] = "\n\n";
            } 

            shared_mem_wrapper->rpc_id = 0;
            shared_mem_wrapper->num_args = 1;
            shared_mem_wrapper->server_send_rpc_call_result();
        }
    }

    string PyMultiprocessingThtsEnv::get_initial_state_py_server() const 
    {
        py::handle py_get_initial_state_fn = py_thts_env->attr("get_initial_state");
        py::object py_init_state = py_get_initial_state_fn();
        return pickle_wrapper->serialise(py_init_state);
    }
    
    shared_ptr<const PyState> PyMultiprocessingThtsEnv::get_initial_state() const 
    {
        shared_mem_wrapper->rpc_id = RPC_get_initial_state;
        shared_mem_wrapper->num_args = 0;
        shared_mem_wrapper->make_rpc_call();
        return make_shared<const PyState>(pickle_wrapper, make_shared<string>(shared_mem_wrapper->args[0]));
    }

    string PyMultiprocessingThtsEnv::is_sink_state_py_server(string& state) const 
    {
        py::object py_state = pickle_wrapper->deserialise(state);
        py::handle py_is_sink_state_fn = py_thts_env->attr("is_sink_state");
        bool is_sink_state = py_is_sink_state_fn(py_state).cast<bool>();
        return is_sink_state ? "T" : "F";
    }

    bool PyMultiprocessingThtsEnv::is_sink_state(shared_ptr<const PyState> state, ThtsEnvContext& ctx) const 
    {
        shared_mem_wrapper->rpc_id = RPC_is_sink_state;
        shared_mem_wrapper->num_args = 1;
        shared_mem_wrapper->args[0] = *state->get_serialised_state();
        shared_mem_wrapper->make_rpc_call();
        return shared_mem_wrapper->args[0] == "T";
    }

    string PyMultiprocessingThtsEnv::get_valid_actions_py_server(string& state) const 
    {
        py::object py_state = pickle_wrapper->deserialise(state);
        py::handle py_get_valid_actions_fn = py_thts_env->attr("get_valid_actions");
        py::object py_valid_actions_list = py_get_valid_actions_fn(py_state);
        return pickle_wrapper->serialise(py_valid_actions_list);
    }
    
    shared_ptr<PyActionVector> PyMultiprocessingThtsEnv::get_valid_actions(
        shared_ptr<const PyState> state, ThtsEnvContext& ctx) const 
    { 
        shared_mem_wrapper->rpc_id = RPC_get_valid_actions;
        shared_mem_wrapper->num_args = 1;
        shared_mem_wrapper->args[0] = *state->get_serialised_state();
        shared_mem_wrapper->make_rpc_call();
        py::gil_scoped_acquire acquire;
        py::list py_valid_actions = pickle_wrapper->deserialise(shared_mem_wrapper->args[0]);
        
        shared_ptr<PyActionVector> valid_actions = make_shared<PyActionVector>();
        for (py::handle py_action : py_valid_actions) {
            py::object py_action_object = py::cast<py::object>(py_action);
            valid_actions->push_back(
                make_shared<const PyAction>(pickle_wrapper, make_shared<py::object>(py_action_object)));
        }
        return valid_actions;
    }

    string PyMultiprocessingThtsEnv::get_transition_distribution_py_server(string& state, string& action) const 
    {
        py::object py_state = pickle_wrapper->deserialise(state);
        py::object py_action = pickle_wrapper->deserialise(action);
        py::handle py_get_transition_distribution_fn = py_thts_env->attr("get_transition_distribution");
        py::object py_transition_prob_map = py_get_transition_distribution_fn(py_state, py_action);
        return pickle_wrapper->serialise(py_transition_prob_map);
    }

    shared_ptr<PyStateDistr> PyMultiprocessingThtsEnv::get_transition_distribution(
        shared_ptr<const PyState> state, shared_ptr<const PyAction> action, ThtsEnvContext& ctx) const 
    {
        shared_mem_wrapper->rpc_id = RPC_get_transition_distribution;
        shared_mem_wrapper->num_args = 2;
        shared_mem_wrapper->args[0] = *state->get_serialised_state();
        shared_mem_wrapper->args[1] = *action->get_serialised_action();
        shared_mem_wrapper->make_rpc_call();
        py::gil_scoped_acquire acquire;
        py::dict py_transition_prob_map = pickle_wrapper->deserialise(shared_mem_wrapper->args[0]);

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

    string PyMultiprocessingThtsEnv::sample_transition_distribution_py_server(string& state, string& action) const
    {
        py::object py_state = pickle_wrapper->deserialise(state);
        py::object py_action = pickle_wrapper->deserialise(action);
        py::handle py_sample_transition_distribution_fn = py_thts_env->attr("sample_transition_distribution");
        py::object py_next_state = py_sample_transition_distribution_fn(py_state, py_action);
        return pickle_wrapper->serialise(py_next_state);
    }

    shared_ptr<const PyState> PyMultiprocessingThtsEnv::sample_transition_distribution(
        shared_ptr<const PyState> state, 
        shared_ptr<const PyAction> action, 
        RandManager& rand_manager, 
        ThtsEnvContext& ctx) const 
    {
        shared_mem_wrapper->rpc_id = RPC_sample_transition_distribution;
        shared_mem_wrapper->num_args = 2;
        shared_mem_wrapper->args[0] = *state->get_serialised_state();
        shared_mem_wrapper->args[1] = *action->get_serialised_action();
        shared_mem_wrapper->make_rpc_call();
        return make_shared<const PyState>(pickle_wrapper, make_shared<string>(shared_mem_wrapper->args[0]));
    }

    string PyMultiprocessingThtsEnv::get_reward_py_server(string& state, string& action) const 
    {
        py::object py_state = pickle_wrapper->deserialise(state);
        py::object py_action = pickle_wrapper->deserialise(action);
        py::handle py_get_reward_fn = py_thts_env->attr("get_reward");
        py::object py_reward = py_get_reward_fn(py_state, py_action);
        return pickle_wrapper->serialise(py_reward);
    }

    double PyMultiprocessingThtsEnv::get_reward(
        shared_ptr<const PyState> state, 
        shared_ptr<const PyAction> action, 
        ThtsEnvContext& ctx) const 
    {
        shared_mem_wrapper->rpc_id = RPC_get_reward;
        shared_mem_wrapper->num_args = 2;
        shared_mem_wrapper->args[0] = *state->get_serialised_state();
        shared_mem_wrapper->args[1] = *action->get_serialised_action();
        shared_mem_wrapper->make_rpc_call();
        py::gil_scoped_acquire acquire;
        return pickle_wrapper->deserialise(shared_mem_wrapper->args[0]).cast<double>();
    }

    void PyMultiprocessingThtsEnv::reset_py_server() const 
    {
        py::handle py_reset_fn = py_thts_env->attr("reset");
        py_reset_fn();
    }

    void PyMultiprocessingThtsEnv::reset() const
    {
        shared_mem_wrapper->rpc_id = RPC_reset;
        shared_mem_wrapper->num_args = 0;
        shared_mem_wrapper->make_rpc_call();
    }
}



/**
 * Boilerplate defined functions. Copied from thts_env_template.h.
 * 
 * TODO: decide if need to write a custom version of these depending on if need partial observability or if need 
 * custom contexts.
 */
namespace thts::python {
    shared_ptr<PyObservationDistr> PyMultiprocessingThtsEnv::get_observation_distribution(
        shared_ptr<const PyAction> action, shared_ptr<const PyState> next_state, ThtsEnvContext& ctx) const 
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

    shared_ptr<const PyObservation> PyMultiprocessingThtsEnv::sample_observation_distribution(
        shared_ptr<const PyAction> action, 
        shared_ptr<const PyState> next_state, 
        RandManager& rand_manager, 
        ThtsEnvContext& ctx) const 
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
    
    shared_ptr<const State> PyMultiprocessingThtsEnv::get_initial_state_itfc() const {
        shared_ptr<const PyState> init_state = get_initial_state();
        return static_pointer_cast<const State>(init_state);
    }

    bool PyMultiprocessingThtsEnv::is_sink_state_itfc(shared_ptr<const State> state, ThtsEnvContext& ctx) const {
        ThtsEnvContext& py_ctx = (ThtsEnvContext&) ctx;
        shared_ptr<const PyState> state_itfc = static_pointer_cast<const PyState>(state);
        return is_sink_state(state_itfc, py_ctx);
    }

    shared_ptr<ActionVector> PyMultiprocessingThtsEnv::get_valid_actions_itfc(shared_ptr<const State> state, ThtsEnvContext& ctx) const {
        ThtsEnvContext& py_ctx = (ThtsEnvContext&) ctx;
        shared_ptr<const PyState> state_itfc = static_pointer_cast<const PyState>(state);
        shared_ptr<vector<shared_ptr<const PyAction>>> valid_actions_itfc = get_valid_actions(state_itfc, py_ctx);

        shared_ptr<ActionVector> valid_actions = make_shared<ActionVector>();
        for (shared_ptr<const PyAction> act : *valid_actions_itfc) {
            valid_actions->push_back(static_pointer_cast<const Action>(act));
        }
        return valid_actions;
    }

    shared_ptr<StateDistr> PyMultiprocessingThtsEnv::get_transition_distribution_itfc(
        shared_ptr<const State> state, shared_ptr<const Action> action, ThtsEnvContext& ctx) const 
    {
        ThtsEnvContext& py_ctx = (ThtsEnvContext&) ctx;
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

    shared_ptr<const State> PyMultiprocessingThtsEnv::sample_transition_distribution_itfc(
       shared_ptr<const State> state, 
       shared_ptr<const Action> action, 
       RandManager& rand_manager, 
       ThtsEnvContext& ctx) const 
    {
        ThtsEnvContext& py_ctx = (ThtsEnvContext&) ctx;
        shared_ptr<const PyState> state_itfc = static_pointer_cast<const PyState>(state);
        shared_ptr<const PyAction> action_itfc = static_pointer_cast<const PyAction>(action);
        shared_ptr<const PyState> obsv = sample_transition_distribution(state_itfc, action_itfc, rand_manager, py_ctx);
        return static_pointer_cast<const State>(obsv);
    }

    shared_ptr<ObservationDistr> PyMultiprocessingThtsEnv::get_observation_distribution_itfc(
        shared_ptr<const Action> action, shared_ptr<const State> next_state, ThtsEnvContext& ctx) const
    {
        ThtsEnvContext& py_ctx = (ThtsEnvContext&) ctx;
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

    shared_ptr<const Observation> PyMultiprocessingThtsEnv::sample_observation_distribution_itfc(
        shared_ptr<const Action> action, 
        shared_ptr<const State> next_state,
         RandManager& rand_manager, 
         ThtsEnvContext& ctx) const
    {
        ThtsEnvContext& py_ctx = (ThtsEnvContext&) ctx;
        shared_ptr<const PyAction> act_itfc = static_pointer_cast<const PyAction>(action);
        shared_ptr<const PyState> next_state_itfc = static_pointer_cast<const PyState>(next_state);
        shared_ptr<const PyObservation> obsv_itfc = sample_observation_distribution(
            act_itfc, next_state_itfc, rand_manager, py_ctx);
        return static_pointer_cast<const Observation>(obsv_itfc);
    }

    double PyMultiprocessingThtsEnv::get_reward_itfc(
        shared_ptr<const State> state, 
        shared_ptr<const Action> action, 
        ThtsEnvContext& ctx) const
    {
        ThtsEnvContext& py_ctx = (ThtsEnvContext&) ctx;
        shared_ptr<const PyState> state_itfc = static_pointer_cast<const PyState>(state);
        shared_ptr<const PyAction> action_itfc = static_pointer_cast<const PyAction>(action);
        return get_reward(state_itfc, action_itfc, py_ctx); 
    }

    void PyMultiprocessingThtsEnv::reset_itfc() const
    {
        reset();
    }
}