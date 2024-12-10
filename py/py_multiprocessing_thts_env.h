#pragma once

#include "thts_env.h"
#include "thts_env_context.h"
#include "thts_manager.h"
#include "py/pickle_wrapper.h"
#include "py/py_thts_types.h"
#include "py/shared_mem_wrapper.h"

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <unistd.h>


namespace thts::python {
    // PyBind
    using namespace thts;
    namespace py = pybind11;

    // ID to identify this env for server processes
    static std::string PY_ENV_SERVER_ID = "py_mp_env"; 

    // Enum for function calls
    enum ThtsEnvRpcFn {
        RPC_kill_server = 0,
        RPC_get_initial_state = 1,
        RPC_is_sink_state = 2,
        RPC_get_valid_actions = 3,
        RPC_get_transition_distribution = 4,
        RPC_sample_transition_distribution = 5,
        RPC_get_reward = 6,
        RPC_reset = 7,
    };

    // Typedef 
    typedef std::unordered_map<std::shared_ptr<const PyState>,double> PyStateDistr;
    typedef std::vector<std::shared_ptr<const PyAction>> PyActionVector;
    typedef std::unordered_map<std::shared_ptr<const PyObservation>,double> PyObservationDistr;

    /** 
     * A ThtsEnv subclass used as a wrapper around an environment defined in python
     * Assumes that the python environment is a subclass the 'PyThtsEnv' (python class defined in py_thts_env.py
     * 
     * Member variables:
     *      py_thts_env: 
     *          A pybind11 py::object pointing to the python implementation an environment
     *      pickle_wrapper:
     *          Pointer to a PickleWrapper for translating between C++ strings and py::objects using Python's pickle
     *      thts_unique_filename:
     *          A filename that is unique for this instance of THTS. Required for 'shared_mem_wrapper' to work.
     *      shared_mem_wrapper:
     *          Class that contains all of the logic to create, manage and interact with shared memory for 
     *          inter process comms.
     *      shared_memory_size_in_bytes:
     *          The size of the shared memory segment to use in the shaed_mem_wrapper (default = 1Mb)
     *      child_pid:
     *          The pid of the child (server) process spawned
     *      module_name:
     *          See class_name
     *      class_name:
     *          If you would import you python env using "from <module_name> import <class_name>", then the strings 
     *          module_name and class_name are needed to make the Thts env.
     *      constructor_kw_args:
     *          A pointer to a py::dict object that specifys the key-word arguments to construct the class
     *      is_server_process:
     *          A boolean to identify if we are currently running in a client process (running thts trials) or a 
     *          server process (interacting with the underlying python environment)
     * 
     * N.B. on the python side, the ThtsEnv needs to be able to take arguments as strings, as arguments are passed to 
     * subprocesses as strings. 
     *      
     */
    class PyMultiprocessingThtsEnv : virtual public ThtsEnv {

        /**
         * Core PyMultiprocessingThtsEnv implementaion.
         */
        protected:
            std::shared_ptr<py::object> py_thts_env;
            std::shared_ptr<PickleWrapper> pickle_wrapper;
            std::string thts_unique_filename;
            std::shared_ptr<SharedMemWrapper> shared_mem_wrapper;
            size_t shared_memory_size_in_bytes;
            pid_t child_pid;

            std::string module_name;
            std::string class_name;
            std::shared_ptr<py::dict> constructor_kw_args;
            bool is_server_process;

        /**
         * Core ThtsEnv implementation functinos.
         */
        protected:
            /**
             * Constructor, passing python object directly
             */
            PyMultiprocessingThtsEnv(
                std::shared_ptr<PickleWrapper> pickle_wrapper,
                std::string& thts_unique_filename,
                std::shared_ptr<py::object> py_thts_env,
                bool is_server_process=false,
                size_t shared_memory_size_in_bytes=1024*1024);

            /**
             * Constructor, passing python module name, class name, and constructor args
             */
        public:
            PyMultiprocessingThtsEnv(
                std::shared_ptr<PickleWrapper> pickle_wrapper,
                std::string& thts_unique_filename,
                std::string module_name,
                std::string class_name,
                std::shared_ptr<py::dict> constructor_kw_args,
                bool is_server_process=false,
                size_t shared_memory_size_in_bytes=1024*1024);

            /**
             * Private copy constructor to implement 
            */
            PyMultiprocessingThtsEnv(PyMultiprocessingThtsEnv& other);

            /**
             * Clone - virtual copy constructor idiom
            */
            virtual std::shared_ptr<ThtsEnv> clone() override;

            /**
             * Give ability to force clean unix shared memory and semaphores
            */
            void clear_unix_sem_and_shm();

            /**
             * Mark destructor as virtual for subclassing.
             */
            virtual ~PyMultiprocessingThtsEnv();

            /**
             * Starts python server process and sets up 'shared_mem_wrapper'
             * 'tid' for the thread that will use this env
             * 
             * Called from client process
            */
            void start_python_server(int tid);

            /**
             * Main function run by the python server process
            */
            void server_main(int tid);

            /**
             * Adds the arguments needed in to run the "py_env_server" program for this env.
             */
            virtual std::string get_multiprocessing_env_type_id();
            virtual void fill_multiprocessing_args(std::vector<std::string>& args, int tid);

            /**
             * Returns the initial state for the environment.
             * 
             * Returns:
             *      Initial state for this environment instance
             */
            std::shared_ptr<std::vector<std::string>> get_initial_state_py_server() const;
            std::shared_ptr<const PyState> get_initial_state() const;

            /**
             * Returns if a state is a sink state.
             * 
             * Args:
             *      state: The state to be checked if it is a sink state
             * 
             * Returns:
             *      True if 'state' is a sink state and false otherwise
             */
            std::shared_ptr<std::vector<std::string>> is_sink_state_py_server(std::string& state) const;
            bool is_sink_state(std::shared_ptr<const PyState> state, ThtsEnvContext& ctx) const;

            /**
             * Returns a list of actions that are valid in a given state.
             * 
             * Args:
             *      state: The state that we want a list of available actions from
             * 
             * Returns:
             *      Returns a list of actions available from 'state'
             */
            std::shared_ptr<std::vector<std::string>> get_valid_actions_py_server(std::string& state) const;
            std::shared_ptr<PyActionVector> get_valid_actions(
                std::shared_ptr<const PyState> state, ThtsEnvContext& ctx) const;

            /**
             * Returns a distribution over successor states from a state action pair.
             * 
             * Given a state and action returns a distribution of possible successor states. The probability 
             * distribution is returned in the form of a map, where the keys are of the State type, and the values are 
             * doubles, which sum to one.
             * 
             * Args:
             *      state: The state to get a transition distribution from
             *      action: The action to get a transition distribution for
             * 
             * Returns:
             *      Returns a successor state distribution from taking 'action' in state 'state'.
             */
            std::shared_ptr<std::unordered_map<std::string,double>> get_transition_distribution_py_server(
                std::string& state, std::string& action) const;
            std::shared_ptr<PyStateDistr> get_transition_distribution(
                std::shared_ptr<const PyState> state, 
                std::shared_ptr<const PyAction> action, 
                ThtsEnvContext& ctx) const;

            /**
             * Samples an successor state when taking an action from a state.
             * 
             * Given a state, action pair, samples a possible successor state that can arrise.
             * 
             * Args:
             *      state: The state to sample an observation from
             *      action: The action taken to sample an observation for
             *      rand_manager: A RandManager ref to access the random number sampling interface
             * 
             * Returns:
             *      Returns an successor state sampled from taking 'action' from 'state'
             */
            std::shared_ptr<std::vector<std::string>> sample_transition_distribution_py_server(
                std::string& state, std::string& action) const;
            std::shared_ptr<const PyState> sample_transition_distribution(
                std::shared_ptr<const PyState> state, 
                std::shared_ptr<const PyAction> action, 
                RandManager& rand_manager, 
                ThtsEnvContext& ctx) const;
            
            /**
             * Returns the reward for a given state, action, observation tuple.
             * 
             * Commonly the reward is written as a function of just the state and action pair. But we provide the 
             * option to depend on the observation too. 
             * 
             * Args:
             *      state: The current state to get a reward for
             *      action: The action taken to get a reward for
             *      observation: 
             *          The (optional) observation sampled from the state, action pair that can optionally be used as 
             *          part of the reward function.
             * 
             * Returns:
             *      The reward for taking 'action' from 'state' (and sampling 'observation')
             */
            virtual std::shared_ptr<std::vector<double>> get_reward_py_server(
                std::string& state, std::string& action) const;
            double get_reward(
                std::shared_ptr<const PyState> state, 
                std::shared_ptr<const PyAction> action, 
                ThtsEnvContext& ctx) const;

            /**
             * Samples a context that can be used to store information throughout a single trial.
             * 
             * Sometimes it is useful to place each trial in some sort of context, or a context can be used to cache 
             * information that doesn't need to be stored in the tree search permenantly, but is useful computationally. 
             * This function generates a context to be used. Most of the time it will be something like an empty map. 
             * 
             * Args:
             *      state: The initial state
             * 
             * Returns:
             *      A ThtsEnvContext object, that will be passed to the Thts functions for a single trial, used to 
             *      provide some context or space for caching.
             */
            void reset_py_server() const;
            void reset() const;



        /**
         * Boilerplate functinos (defined in thts_env_template.{h,cpp}) using the default implementations provided by 
         * thts_env.{h,cpp}. 
         */
        public:
            /**
             * Returns a distribution over observations from a (next) state, action pair.
             * 
             * Given a state and action returns a distribution of possible observations. The probability 
             * distribution is returned in the form of a map, where the keys are of the Observation type, and the 
             * values are doubles, which sum to one.
             * 
             * A default implementation is provided for full observable environments, where observation == next state.
             * 
             * Args:
             *      action: The action to get an observation distribution for
             *      next_state: The state (arriving in)  to get an observation distribution from
             * 
             * Returns:
             *      Returns a distribution over observations from taking 'action' in state 'state'.
             */
            virtual std::shared_ptr<PyObservationDistr> get_observation_distribution(
                std::shared_ptr<const PyAction> action, 
                std::shared_ptr<const PyState> next_state, 
                ThtsEnvContext& ctx) const;

            /**
             * Samples an observation when arriving in a (next) state after taking an action.
             * 
             * Given a state-action pair, samples a possible sobservation.
             * 
             * A default implementation is provided for full observable environments, where observation == next state.
             * 
             * Args:
             *      action: The action taken to sample an observation for
             *      next_state: The state (arriving in)  to sample an observation for
             *      rand_manager: A RandManager ref to access the random number sampling interface
             * 
             * Returns:
             *      Returns an observation sampled from taking 'action' that arived in 'next_state'
             */
            virtual std::shared_ptr<const PyObservation> sample_observation_distribution(
                std::shared_ptr<const PyAction> action, 
                std::shared_ptr<const PyState> next_state, 
                RandManager& rand_manager, 
                ThtsEnvContext& ctx) const;



        /**
         * ThtsEnv interface function definitions. Boilerplate implementations provided from thts_env_template.h
         */
        public:
            virtual std::shared_ptr<const State> get_initial_state_itfc() const override;
            virtual bool is_sink_state_itfc(std::shared_ptr<const State> state, ThtsEnvContext& ctx) const override;
            virtual std::shared_ptr<ActionVector> get_valid_actions_itfc(
                std::shared_ptr<const State> state, ThtsEnvContext& ctx) const override;
            virtual std::shared_ptr<StateDistr> get_transition_distribution_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action, 
                ThtsEnvContext& ctx) const override;
            virtual std::shared_ptr<const State> sample_transition_distribution_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action, 
                 RandManager& rand_manager, 
                 ThtsEnvContext& ctx) const override;
            virtual std::shared_ptr<ObservationDistr> get_observation_distribution_itfc(
                std::shared_ptr<const Action> action, 
                std::shared_ptr<const State> next_state, 
                ThtsEnvContext& ctx) const override;
            virtual std::shared_ptr<const Observation> sample_observation_distribution_itfc(
                std::shared_ptr<const Action> action, 
                std::shared_ptr<const State> next_state, 
                 RandManager& rand_manager, 
                 ThtsEnvContext& ctx) const override;
            virtual double get_reward_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action, 
                ThtsEnvContext& ctx) const override;
            virtual void reset_itfc() const override;
        
        /**
         * Implemented in thts_env.{h,cpp}
         */
        // public:
        //     bool is_fully_observable();
    };
}



