// #pragma once

// #include "thts_env.h"
// #include "thts_env_context.h"
// #include "thts_manager.h"
// #include "py/pickle_wrapper.h"
// #include "py/py_thts_types.h"

// #include <pybind11/pybind11.h>
// #include <pybind11/embed.h>

// #include <memory>
// #include <mutex>
// #include <string>
// #include <unordered_map>
// #include <vector>


// namespace thts::python {
//     // PyBind
//     using namespace thts;
//     namespace py = pybind11;

//     // Typedef 
//     typedef std::unordered_map<std::shared_ptr<const PyState>,double> PyStateDistr;
//     typedef std::vector<std::shared_ptr<const PyAction>> PyActionVector;
//     typedef std::unordered_map<std::shared_ptr<const PyObservation>,double> PyObservationDistr;

//     /** 
//      * NOTE:
//      * - this version of PyThtsEnv will be slow, and wont gain much from multi-threading, as there is lots of gil 
//      *      locking happening in this class. If want to interact with a Python env from multiple threads then use 
//      *      PyMultiprocessingThtsEnv
//      * 
//      * 
//      * A ThtsEnv subclass used as a wrapper around an environment defined in python
//      * Assumes that the python environment is a subclass the 'PyThtsEnv' (python class defined in py_thts_env.py
//      * 
//      * This class allows for safe use of a single copy of a python env shared between multiple threads
//      * ((ThtsPool) Assumes that python can be run under subinterpreters - numpy code cant!)
//      * 
//      * Protecting python variables:
//      * - py_thts_env needs to be protected by a lock
//      * - if py_thts_env may return the same objects between different function calls then those objects need to be 
//      *      protected from concurrent accesses (so we have a lock per function)
//      * - if not, then we only need to proect using py_thts_env object
//      * 
//      * Member variables:
//      *      multiple_threads_using_this_env:
//      *      py_thts_env: 
//      *          A pybind11 py::object pointing to the python implementation an environment
//      *      py_env_may_return_shared_py_obj: 
//      *          If the py_thts_env pybind object might return the same object between different calls
//      *      lock: 
//      *          For protecting the python objects we're touching
//      */
//     class PyThtsEnv : public ThtsEnv {

//         /**
//          * Core PyThtsEnv implementaion.
//          */
//         protected:
//             std::shared_ptr<py::object> py_thts_env;
//             std::shared_ptr<PickleWrapper> pickle_wrapper;

//         /**
//          * Core ThtsEnv implementation functinos.
//          */
//         public:
//             /**
//              * Constructor
//              */
//             PyThtsEnv(
//                 std::shared_ptr<py::object> py_thts_env, 
//                 std::shared_ptr<PickleWrapper> pickle_wrapper);

//             /**
//              * Private copy constructor to implement 
//             */
//             PyThtsEnv(PyThtsEnv& other);

//             /**
//              * Clone - virtual copy constructor idiom
//             */
//             virtual std::shared_ptr<ThtsEnv> clone() override;

//             /**
//              * Mark destructor as virtual for subclassing.
//              */
//             virtual ~PyThtsEnv();

//             /**
//              * Returns the initial state for the environment.
//              * 
//              * Returns:
//              *      Initial state for this environment instance
//              */
//         public:
//             std::shared_ptr<const PyState> get_initial_state() const;

//             /**
//              * Returns if a state is a sink state.
//              * 
//              * Args:
//              *      state: The state to be checked if it is a sink state
//              * 
//              * Returns:
//              *      True if 'state' is a sink state and false otherwise
//              */
//             bool is_sink_state(std::shared_ptr<const PyState> state, ThtsEnvContext& ctx) const;

//             /**
//              * Returns a list of actions that are valid in a given state.
//              * 
//              * Args:
//              *      state: The state that we want a list of available actions from
//              * 
//              * Returns:
//              *      Returns a list of actions available from 'state'
//              */
//             std::shared_ptr<PyActionVector> get_valid_actions(
//                 std::shared_ptr<const PyState> state, ThtsEnvContext& ctx) const;

//             /**
//              * Returns a distribution over successor states from a state action pair.
//              * 
//              * Given a state and action returns a distribution of possible successor states. The probability 
//              * distribution is returned in the form of a map, where the keys are of the State type, and the values are 
//              * doubles, which sum to one.
//              * 
//              * Args:
//              *      state: The state to get a transition distribution from
//              *      action: The action to get a transition distribution for
//              * 
//              * Returns:
//              *      Returns a successor state distribution from taking 'action' in state 'state'.
//              */
//             std::shared_ptr<PyStateDistr> get_transition_distribution(
//                 std::shared_ptr<const PyState> state, 
//                 std::shared_ptr<const PyAction> action, 
//                 ThtsEnvContext& ctx) const;

//             /**
//              * Samples an successor state when taking an action from a state.
//              * 
//              * Given a state, action pair, samples a possible successor state that can arrise.
//              * 
//              * Args:
//              *      state: The state to sample an observation from
//              *      action: The action taken to sample an observation for
//              *      rand_manager: A RandManager ref to access the random number sampling interface
//              * 
//              * Returns:
//              *      Returns an successor state sampled from taking 'action' from 'state'
//              */
//             std::shared_ptr<const PyState> sample_transition_distribution(
//                 std::shared_ptr<const PyState> state, 
//                 std::shared_ptr<const PyAction> action, 
//                 RandManager& rand_manager, 
//                 ThtsEnvContext& ctx) const;
            
//             /**
//              * Returns the reward for a given state, action, observation tuple.
//              * 
//              * Commonly the reward is written as a function of just the state and action pair. But we provide the 
//              * option to depend on the observation too. 
//              * 
//              * Args:
//              *      state: The current state to get a reward for
//              *      action: The action taken to get a reward for
//              *      observation: 
//              *          The (optional) observation sampled from the state, action pair that can optionally be used as 
//              *          part of the reward function.
//              * 
//              * Returns:
//              *      The reward for taking 'action' from 'state' (and sampling 'observation')
//              */
//             double get_reward(
//                 std::shared_ptr<const PyState> state, 
//                 std::shared_ptr<const PyAction> action, 
//                 ThtsEnvContext& ctx) const;

//             /**
//              * Samples a context that can be used to store information throughout a single trial.
//              * 
//              * Sometimes it is useful to place each trial in some sort of context, or a context can be used to cache 
//              * information that doesn't need to be stored in the tree search permenantly, but is useful computationally. 
//              * This function generates a context to be used. Most of the time it will be something like an empty map. 
//              * 
//              * Args:
//              *      state: The initial state
//              * 
//              * Returns:
//              *      A ThtsEnvContext object, that will be passed to the Thts functions for a single trial, used to 
//              *      provide some context or space for caching.
//              */
//             virtual void reset() const;



//         /**
//          * Boilerplate functinos (defined in thts_env_template.{h,cpp}) using the default implementations provided by 
//          * thts_env.{h,cpp}. 
//          */
//         public:
//             /**
//              * Returns a distribution over observations from a (next) state, action pair.
//              * 
//              * Given a state and action returns a distribution of possible observations. The probability 
//              * distribution is returned in the form of a map, where the keys are of the Observation type, and the 
//              * values are doubles, which sum to one.
//              * 
//              * A default implementation is provided for full observable environments, where observation == next state.
//              * 
//              * Args:
//              *      action: The action to get an observation distribution for
//              *      next_state: The state (arriving in)  to get an observation distribution from
//              * 
//              * Returns:
//              *      Returns a distribution over observations from taking 'action' in state 'state'.
//              */
//             virtual std::shared_ptr<PyObservationDistr> get_observation_distribution(
//                 std::shared_ptr<const PyAction> action, 
//                 std::shared_ptr<const PyState> next_state, 
//                 ThtsEnvContext& ctx) const;

//             /**
//              * Samples an observation when arriving in a (next) state after taking an action.
//              * 
//              * Given a state-action pair, samples a possible sobservation.
//              * 
//              * A default implementation is provided for full observable environments, where observation == next state.
//              * 
//              * Args:
//              *      action: The action taken to sample an observation for
//              *      next_state: The state (arriving in)  to sample an observation for
//              *      rand_manager: A RandManager ref to access the random number sampling interface
//              * 
//              * Returns:
//              *      Returns an observation sampled from taking 'action' that arived in 'next_state'
//              */
//             virtual std::shared_ptr<const PyObservation> sample_observation_distribution(
//                 std::shared_ptr<const PyAction> action, 
//                 std::shared_ptr<const PyState> next_state, 
//                 RandManager& rand_manager, 
//                 ThtsEnvContext& ctx) const;



//         /**
//          * ThtsEnv interface function definitions. Boilerplate implementations provided from thts_env_template.h
//          */
//         public:
//             virtual std::shared_ptr<const State> get_initial_state_itfc() const override;
//             virtual bool is_sink_state_itfc(std::shared_ptr<const State> state, ThtsEnvContext& ctx) const override;
//             virtual std::shared_ptr<ActionVector> get_valid_actions_itfc(
//                 std::shared_ptr<const State> state, ThtsEnvContext& ctx) const override;
//             virtual std::shared_ptr<StateDistr> get_transition_distribution_itfc(
//                 std::shared_ptr<const State> state, 
//                 std::shared_ptr<const Action> action, 
//                 ThtsEnvContext& ctx) const override;
//             virtual std::shared_ptr<const State> sample_transition_distribution_itfc(
//                 std::shared_ptr<const State> state, 
//                 std::shared_ptr<const Action> action, 
//                  RandManager& rand_manager, 
//                  ThtsEnvContext& ctx) const override;
//             virtual std::shared_ptr<ObservationDistr> get_observation_distribution_itfc(
//                 std::shared_ptr<const Action> action, 
//                 std::shared_ptr<const State> next_state, 
//                 ThtsEnvContext& ctx) const override;
//             virtual std::shared_ptr<const Observation> sample_observation_distribution_itfc(
//                 std::shared_ptr<const Action> action, 
//                 std::shared_ptr<const State> next_state, 
//                  RandManager& rand_manager, 
//                  ThtsEnvContext& ctx) const override;
//             virtual double get_reward_itfc(
//                 std::shared_ptr<const State> state, 
//                 std::shared_ptr<const Action> action, 
//                 ThtsEnvContext& ctx) const override;
//             virtual void reset_itfc() const override;
        
//         /**
//          * Implemented in thts_env.{h,cpp}
//          */
//         // public:
//         //     bool is_fully_observable();
//     };
// }



