/**
 * Template for ThtsEnv subclasses, because it involves some boilerplate code that will generally look the same.
 * 
 * To use the template, copy the relevant sections into your .h and .cpp files, and make the following find and replace
 * operations:
 *      _Env -> YourEnvClass
 *      _S -> YourStateClass
 *      _A -> YourActionClass
 *      _O -> YourObservationClass
 * 
 * Finally, complete all of the TODO comments inline.
 */

/**
 * -----------------------------------
 * .h template - copy into .h file
 * -----------------------------------
 */

#pragma once

#include "thts_env.h"
#include "thts_types.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// TODO: any other includes needed

namespace thts{
    // TODO: delete these forward declarations (added to stop IDEs showing compile errors).
    class _S;
    class _A;
    class _O;

    /** 
     * TODO: write your docstring
     */
    class _Env : public ThtsEnv {

        // TODO: add an id for your env
        protected:
            std::string env_id = "my_thts_env_id";

        public:
            /**
             * TODO: Core THTS implementation functions. Implement in .cpp and add any docstrings.
             */
            _Env();
            virtual ~_Env() = default;

            std::shared_ptr<const _S> get_initial_state() const;

            bool is_sink_state(std::shared_ptr<const _S> state) const;

            std::shared_ptr<std::vector<std::shared_ptr<const _A>>> get_valid_actions(std::shared_ptr<const _S> state) const;

            std::shared_ptr<std::unordered_map<std::shared_ptr<const _O>,double>> get_transition_distribution(
                std::shared_ptr<const _S> state, std::shared_ptr<const _A> action) const;

            std::shared_ptr<const _O> sample_transition_distribution(
                std::shared_ptr<const _S> state, std::shared_ptr<const _A> action) const;

            double get_reward(
                std::shared_ptr<const _S> state, 
                std::shared_ptr<const _A> action, 
                std::shared_ptr<const _O> observation=nullptr) const;

            /**
             * ThtsEnv interface function definitions. Boilerplate implementations provided from thts_env_template.h
             */
            virtual std::shared_ptr<const State> get_initial_state_itfc() const;
            virtual bool is_sink_state_itfc(std::shared_ptr<const State> state) const;
            virtual std::shared_ptr<ActionVector> get_valid_actions_itfc(std::shared_ptr<const State> state) const;
            virtual std::shared_ptr<ObservationDistr> get_transition_distribution_itfc(
                std::shared_ptr<const State> state, std::shared_ptr<const Action> action) const;
            virtual std::shared_ptr<const Observation> sample_transition_distribution_itfc(
                std::shared_ptr<const State> state, std::shared_ptr<const Action> action) const;
            virtual double get_reward_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action, 
                std::shared_ptr<const Observation> observation=nullptr) const;
    };
}



/**
 * -----------------------------------
 * .cpp template - copy into .cpp file
 * -----------------------------------
 */

// TODO: add include for your header file

using namespace std; 

/**
 * TODO: implement your class here.
 */
namespace thts {
    _Env::_Env() {

    }

    shared_ptr<const _S> _Env::get_initial_state() const {
        return nullptr;
    }

    bool _Env::is_sink_state(shared_ptr<const _S> state) const {
        return false;
    }

    shared_ptr<vector<shared_ptr<const _A>>> _Env::get_valid_actions(shared_ptr<const _S> state) const {
        return nullptr;
    }

    shared_ptr<unordered_map<shared_ptr<const _O>,double>> _Env::get_transition_distribution(
        shared_ptr<const _S> state, shared_ptr<const _A> action) const 
    {
        return nullptr;
    }

    shared_ptr<const _O> _Env::sample_transition_distribution(
        shared_ptr<const _S> state, shared_ptr<const _A> action) const 
    {
        return nullptr;
    }

    double _Env::get_reward(
        shared_ptr<const _S> state, 
        shared_ptr<const _A> action, 
        shared_ptr<const _O> observation=nullptr) const 
    {
        return 0.0;
    }
}

/**
 * Boilerplate ThtsEnv interface implementation. Copied from thts_env_template.h.
 * All this code basically calls the corresponding implementation function, with approprtiate casts before/after.
 */
namespace thts {
    
    shared_ptr<const State> _Env::get_initial_state_itfc() const {
        shared_ptr<const _S> init_state = get_initial_state();
        return static_pointer_cast<const State>(init_state);
    }

    bool _Env::is_sink_state_itfc(shared_ptr<const State> state) const {
        shared_ptr<const _S> state_itfc = static_pointer_cast<const _S>(state);
        return is_sink_state(state_itfc);
    }

    shared_ptr<ActionVector> _Env::get_valid_actions_itfc(shared_ptr<const State> state) const {
        shared_ptr<const _S> state_itfc = static_pointer_cast<const _S>(state);
        shared_ptr<vector<shared_ptr<const _A>>> valid_actions_itfc = get_valid_actions(state_itfc);

        shared_ptr<ActionVector> valid_actions = make_shared<ActionVector>();
        for (shared_ptr<const _A> act : *valid_actions_itfc) {
            valid_actions->push_back(static_pointer_cast<const Action>(act));
        }
        return valid_actions;
    }

    shared_ptr<ObservationDistr> _Env::get_transition_distribution_itfc(
        shared_ptr<const State> state, shared_ptr<const Action> action) const 
    {
        shared_ptr<const _S> state_itfc = static_pointer_cast<const _S>(state);
        shared_ptr<const _A> action_itfc = static_pointer_cast<const _A>(action);
        shared_ptr<unordered_map<shared_ptr<const _O>,double>> distr_itfc = get_transition_distribution(
            state_itfc, action_itfc);
        
        shared_ptr<ObservationDistr> distr = make_shared<ObservationDistr>(); 
        for (pair<shared_ptr<const _O>,double> key_val_pair : *distr_itfc) {
            shared_ptr<const Observation> obsv = static_pointer_cast<const Observation>(key_val_pair.first);
            double prob = key_val_pair.second;
            distr->insert_or_assign(obsv, prob);
        }
        return distr;
    }

    shared_ptr<const Observation> _Env::sample_transition_distribution_itfc(
        shared_ptr<const State> state, shared_ptr<const Action> action) const 
    {
        shared_ptr<const _S> state_itfc = static_pointer_cast<const _S>(state);
        shared_ptr<const _A> action_itfc = static_pointer_cast<const _A>(action);
        shared_ptr<const _O> obsv = sample_transition_distribution(state_itfc, action_itfc);
        return static_pointer_cast<const Observation>(obsv);
    }

    double _Env::get_reward_itfc(
        shared_ptr<const State> state, 
        shared_ptr<const Action> action, 
        shared_ptr<const Observation> observation=nullptr) const
    {
        shared_ptr<const _S> state_itfc = static_pointer_cast<const _S>(state);
        shared_ptr<const _A> action_itfc = static_pointer_cast<const _A>(action);
        shared_ptr<const _O> obsv_itfc = static_pointer_cast<const _O>(observation);
        return get_reward(state_itfc, action_itfc, obsv_itfc);
    }
}