#include "toy_envs/d_chain_env.h"

using namespace std; 

namespace thts {
    /**
     * Construct
    */
    DChainEnv::DChainEnv(int D, double final_reward) : 
        ThtsEnv(true), 
        D(D), 
        final_reward(final_reward), 
        cached_actions(make_shared<IntActionVector>())
    {
        cached_actions->push_back(make_shared<IntAction>(DCHAIN_RIGHT));
        cached_actions->push_back(make_shared<IntAction>(DCHAIN_DOWN));
    }

    /**
     * Init state is at zero
    */
    shared_ptr<const IntState> DChainEnv::get_initial_state() const {
        return make_shared<const IntState>(0);
    }

    /**
     * -1 used for sink state when move down, D is end of chain
    */
    bool DChainEnv::is_sink_state(shared_ptr<const IntState> state) const {
        return state->state == -1 || state->state == D;
    }

    /**
     * Can always move right or down, unless in sink (have moved down or got to end)
    */
    shared_ptr<IntActionVector> DChainEnv::get_valid_actions(shared_ptr<const IntState> state) const {
        if (is_sink_state(state)) return make_shared<IntActionVector>();
        return cached_actions;
    }

    /**
     * Distr = next state with prob 1
    */
    shared_ptr<IntStateDistr> DChainEnv::get_transition_distribution(
        shared_ptr<const IntState> state, shared_ptr<const IntAction> action) const 
    {
        shared_ptr<const IntState> next_state = sample_transition_distribution(state, action);
        shared_ptr<IntStateDistr> next_state_distr = make_shared<IntStateDistr>();
        next_state_distr->insert_or_assign(next_state, 1.0);
        return next_state_distr;
    }

    /**
     * Next state
    */
    shared_ptr<const IntState> DChainEnv::sample_transition_distribution(
        shared_ptr<const IntState> state, shared_ptr<const IntAction> action) const 
    {
        if (action->action == DCHAIN_RIGHT) {
            return make_shared<const IntState>(state->state + 1);
        }
        // if action->action == DHCAIN_DOWN
        return make_shared<const IntState>(-1);
    }

    /**
     * Deterministic env, call the other version of this fn
    */
    shared_ptr<const IntState> DChainEnv::sample_transition_distribution(
        shared_ptr<const IntState> state, shared_ptr<const IntAction> action, RandManager& rand_manager) const 
    {
        return sample_transition_distribution(state, action);
    }

    /**
     * 
    */
    double DChainEnv::get_reward(
        shared_ptr<const IntState> state, 
        shared_ptr<const IntAction> action, 
        shared_ptr<const IntState> observation) const 
    {
        if (action->action == DCHAIN_DOWN) {
            return (D - state->state - 1) / (double) D;
        } else if (action->action == DCHAIN_RIGHT && state->state == D-1) {
            return final_reward;
        }
        return 0.0;
    }
}



/**
 * Boilerplate defined functions. Copied from thts_env_template.h.
 * 
 * TODO: decide if need to write a custom version of these depending on if need partial observability or if need 
 * custom contexts.
 */
namespace thts {
    shared_ptr<IntStateDistr> DChainEnv::get_observation_distribution(
        shared_ptr<const IntAction> action, shared_ptr<const IntState> next_state) const 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<ObservationDistr> distr_itfc = ThtsEnv::get_observation_distribution_itfc(
            act_itfc, next_state_itfc);
        shared_ptr<IntStateDistr> distr;
        for (pair<const shared_ptr<const Observation>,double> pr : *distr_itfc) {
            shared_ptr<const IntState> obsv = static_pointer_cast<const IntState>(pr.first);
            distr->insert_or_assign(obsv, pr.second);
        }
        return distr;
    }

    shared_ptr<const IntState> DChainEnv::sample_observation_distribution(
        shared_ptr<const IntAction> action, 
        shared_ptr<const IntState> next_state, 
        RandManager& rand_manager) const 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<const Observation> obsv_itfc = ThtsEnv::sample_observation_distribution_itfc(
            act_itfc, next_state_itfc, rand_manager);
        return static_pointer_cast<const IntState>(obsv_itfc);
    }

    shared_ptr<ThtsEnvContext> DChainEnv::sample_context(shared_ptr<const IntState> state) const
    {
        shared_ptr<const State> state_itfc = static_pointer_cast<const State>(state);
        shared_ptr<ThtsEnvContext> context = ThtsEnv::sample_context_itfc(state_itfc);
        return static_pointer_cast<ThtsEnvContext>(context);
    }
}



/**
 * Boilerplate ThtsEnv interface implementation. Copied from thts_env_template.h.
 * All this code basically calls the corresponding implementation function, with approprtiate casts before/after.
 */
namespace thts {
    
    shared_ptr<const State> DChainEnv::get_initial_state_itfc() const {
        shared_ptr<const IntState> init_state = get_initial_state();
        return static_pointer_cast<const State>(init_state);
    }

    bool DChainEnv::is_sink_state_itfc(shared_ptr<const State> state) const {
        shared_ptr<const IntState> state_itfc = static_pointer_cast<const IntState>(state);
        return is_sink_state(state_itfc);
    }

    shared_ptr<ActionVector> DChainEnv::get_valid_actions_itfc(shared_ptr<const State> state) const {
        shared_ptr<const IntState> state_itfc = static_pointer_cast<const IntState>(state);
        shared_ptr<vector<shared_ptr<const IntAction>>> valid_actions_itfc = get_valid_actions(state_itfc);

        shared_ptr<ActionVector> valid_actions = make_shared<ActionVector>();
        for (shared_ptr<const IntAction> act : *valid_actions_itfc) {
            valid_actions->push_back(static_pointer_cast<const Action>(act));
        }
        return valid_actions;
    }

    shared_ptr<StateDistr> DChainEnv::get_transition_distribution_itfc(
        shared_ptr<const State> state, shared_ptr<const Action> action) const 
    {
        shared_ptr<const IntState> state_itfc = static_pointer_cast<const IntState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<IntStateDistr> distr_itfc = get_transition_distribution(state_itfc, action_itfc);
        
        shared_ptr<StateDistr> distr = make_shared<StateDistr>(); 
        for (pair<shared_ptr<const IntState>,double> key_val_pair : *distr_itfc) {
            shared_ptr<const State> obsv = static_pointer_cast<const State>(key_val_pair.first);
            double prob = key_val_pair.second;
            distr->insert_or_assign(obsv, prob);
        }
        return distr;
    }

    shared_ptr<const State> DChainEnv::sample_transition_distribution_itfc(
       shared_ptr<const State> state, shared_ptr<const Action> action, RandManager& rand_manager) const 
    {
        shared_ptr<const IntState> state_itfc = static_pointer_cast<const IntState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<const IntState> obsv = sample_transition_distribution(state_itfc, action_itfc, rand_manager);
        return static_pointer_cast<const State>(obsv);
    }

    shared_ptr<ObservationDistr> DChainEnv::get_observation_distribution_itfc(
        shared_ptr<const Action> action, shared_ptr<const State> next_state) const
    {
        shared_ptr<const IntAction> act_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<const IntState> next_state_itfc = static_pointer_cast<const IntState>(next_state);
        shared_ptr<IntStateDistr> distr_itfc = get_observation_distribution(
            act_itfc, next_state_itfc);
        shared_ptr<ObservationDistr> distr;
        for (pair<const shared_ptr<const IntState>,double> pr : *distr_itfc) {
            shared_ptr<const Observation> obsv = static_pointer_cast<const Observation>(pr.first);
            distr->insert_or_assign(obsv, pr.second);
        }
        return distr;
    }     

    shared_ptr<const Observation> DChainEnv::sample_observation_distribution_itfc(
        shared_ptr<const Action> action, 
        shared_ptr<const State> next_state,
         RandManager& rand_manager) const
    {
        shared_ptr<const IntAction> act_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<const IntState> next_state_itfc = static_pointer_cast<const IntState>(next_state);
        shared_ptr<const IntState> obsv_itfc = sample_observation_distribution(
            act_itfc, next_state_itfc, rand_manager);
        return static_pointer_cast<const Observation>(obsv_itfc);
    }

    double DChainEnv::get_reward_itfc(
        shared_ptr<const State> state, 
        shared_ptr<const Action> action, 
        shared_ptr<const Observation> observation) const
    {
        shared_ptr<const IntState> state_itfc = static_pointer_cast<const IntState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<const IntState> obsv_itfc = static_pointer_cast<const IntState>(observation);
        return get_reward(state_itfc, action_itfc, obsv_itfc);
    }

    shared_ptr<ThtsEnvContext> DChainEnv::sample_context_itfc(shared_ptr<const State> state) const
    {
        shared_ptr<const IntState> state_itfc = static_pointer_cast<const IntState>(state);
        shared_ptr<ThtsEnvContext> context = sample_context(state_itfc);
        return static_pointer_cast<ThtsEnvContext>(context);
    }
}