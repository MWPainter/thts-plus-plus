#include "main_mo/envs/tree_env.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

#include <mo/mo_helper.h>

#include <cmath>

using namespace std; 
namespace py = pybind11;


namespace thts {
    ToyTreeEnv::ToyTreeEnv(
        int reward_dim, int num_xtra_actions, double axis_reward_ratio, double random_action_prob) : 
        MoThtsEnv(reward_dim, true),
        total_actions(reward_dim + num_xtra_actions),
        num_xtra_actions(num_xtra_actions),
        reward_scale(axis_reward_ratio),
        horizon(reward_dim * (reward_dim - 1) / 2),
        random_action_prob(random_action_prob),
        xtra_action_rewards(),
        xtra_reward_dims(),
        rand_action_rng_mutex(),
        rand_action_rng(std::random_device{}())
    {
        for (int i=0; i<reward_dim; i++) 
        {
            for (int j=i+1; j<reward_dim; j++) 
            {
                xtra_reward_dims.push_back(make_pair(i,j));
            }
        }

        for (int i=0; i<num_xtra_actions; i++) 
        {
            double angle = 0.5 * M_PI * i / num_xtra_actions;
            xtra_action_rewards.push_back(make_pair(cos(angle),sin(angle)));
        }
    }

    ToyTreeEnv::ToyTreeEnv(const ToyTreeEnv& other) :
        MoThtsEnv(other.reward_dim, true),
        total_actions(other.total_actions),
        num_xtra_actions(other.num_xtra_actions),
        reward_scale(other.reward_scale),
        horizon(other.horizon),
        random_action_prob(other.random_action_prob),
        xtra_action_rewards(other.xtra_action_rewards),
        xtra_reward_dims(other.xtra_reward_dims),
        rand_action_rng_mutex(),
        rand_action_rng(std::random_device{}())
    {
    }

    shared_ptr<ThtsEnv> ToyTreeEnv::clone() {
        return std::dynamic_pointer_cast<ThtsEnv>(std::make_shared<ToyTreeEnv>(*this));
    }

    /**
     * Returns a vector full of zeros
     */
    shared_ptr<const IntState> ToyTreeEnv::get_initial_state() const 
    {
        return make_shared<const IntState>(0);
    }

    bool ToyTreeEnv::is_sink_state(shared_ptr<const IntState> state) const 
    {
        return state->state == horizon;
    }

    shared_ptr<IntActionVector> ToyTreeEnv::get_valid_actions(shared_ptr<const IntState> state) const 
    {
        shared_ptr<IntActionVector> valid_actions = make_shared<IntActionVector>();
        valid_actions->reserve(total_actions);
        for (int i=0; i<total_actions; i++) {
            valid_actions->push_back(make_shared<IntAction>(i));
        }
        return valid_actions;
    }

    shared_ptr<IntState> ToyTreeEnv::get_next_state(
        shared_ptr<const IntState> state, shared_ptr<const IntAction> action) const
    {
        return make_shared<IntState>(state->state + 1);   
    }

    shared_ptr<IntStateDistr> ToyTreeEnv::get_transition_distribution(
        shared_ptr<const IntState> state, shared_ptr<const IntAction> action) const 
    {
        shared_ptr<IntStateDistr> distr;
        shared_ptr<IntState> next_state = get_next_state(state,action);
        distr->insert_or_assign(next_state,1.0);
        return distr;
    }

    shared_ptr<const IntState> ToyTreeEnv::sample_transition_distribution(
        shared_ptr<const IntState> state, shared_ptr<const IntAction> action, RandManager& rand_manager) const 
    {
        return get_next_state(state,action);
    }
 
    Eigen::ArrayXd ToyTreeEnv::get_mo_reward(
        shared_ptr<const IntState> state, 
        shared_ptr<const IntAction> action) const 
    {
        Eigen::ArrayXd reward = Eigen::ArrayXd::Zero(reward_dim);

        int action_idx = action->action;

        // With probability 'random_action_prob', replace the chosen action by a
        // uniformly random action in [0, total_actions). This makes the env stochastic
        // w.r.t. rewards while keeping the (deterministic) state-transition dynamics.
        if (random_action_prob > 0.0f) {
            std::lock_guard<std::mutex> lg(rand_action_rng_mutex);
            std::uniform_real_distribution<double> unit(0.0f, 1.0f);
            if (unit(rand_action_rng) < random_action_prob) {
                std::uniform_int_distribution<int> act_distr(0, total_actions - 1);
                action_idx = act_distr(rand_action_rng);
            }
        }

        if (action_idx < reward_dim) {
            reward[action_idx] = 1.0;
            reward *= reward_scale;
        } else {
            int timestep = state->state;
            int xtra_action_idx = action_idx - reward_dim;
            reward[xtra_reward_dims[timestep].first] = xtra_action_rewards[xtra_action_idx].first;
            reward[xtra_reward_dims[timestep].second] = xtra_action_rewards[xtra_action_idx].second;
        }

        double horizon_scaling = 1.0 / double(horizon);
        reward *= horizon_scaling;
        return reward;
    }
}



/**
 * Boilerplate ThtsEnv interface implementation. Copied from thts_env_template.h.
 * All this code basically calls the corresponding implementation function, with approprtiate casts before/after.
 */
namespace thts {
    
    shared_ptr<const State> ToyTreeEnv::get_initial_state_itfc() const 
    {
        shared_ptr<const IntState> init_state = get_initial_state();
        return static_pointer_cast<const State>(init_state);
    }

    bool ToyTreeEnv::is_sink_state_itfc(shared_ptr<const State> state, ThtsContext& ctx) const 
    {
        shared_ptr<const IntState> state_itfc = static_pointer_cast<const IntState>(state);
        return is_sink_state(state_itfc);
    }

    shared_ptr<ActionVector> ToyTreeEnv::get_valid_actions_itfc(
        shared_ptr<const State> state, ThtsContext& ctx) const 
    {
        shared_ptr<const IntState> state_itfc = static_pointer_cast<const IntState>(state);
        shared_ptr<vector<shared_ptr<const IntAction>>> valid_actions_itfc = get_valid_actions(state_itfc);

        shared_ptr<ActionVector> valid_actions = make_shared<ActionVector>();
        for (shared_ptr<const IntAction> act : *valid_actions_itfc) {
            valid_actions->push_back(static_pointer_cast<const Action>(act));
        }
        return valid_actions;
    }

    shared_ptr<StateDistr> ToyTreeEnv::get_transition_distribution_itfc(
        shared_ptr<const State> state, shared_ptr<const Action> action, ThtsContext& ctx) const 
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

    shared_ptr<const State> ToyTreeEnv::sample_transition_distribution_itfc(
       shared_ptr<const State> state, 
       shared_ptr<const Action> action, 
       RandManager& rand_manager, 
       ThtsContext& ctx) const 
    {
        shared_ptr<const IntState> state_itfc = static_pointer_cast<const IntState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<const IntState> obsv = sample_transition_distribution(state_itfc, action_itfc, rand_manager);
        return static_pointer_cast<const State>(obsv);
    }

    Eigen::ArrayXd ToyTreeEnv::get_mo_reward_itfc(
        shared_ptr<const State> state, 
        shared_ptr<const Action> action,
        ThtsContext& ctx) const
    {
        shared_ptr<const IntState> state_itfc = static_pointer_cast<const IntState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        return get_mo_reward(state_itfc, action_itfc);
    }
}