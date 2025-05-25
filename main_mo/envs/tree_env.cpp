#include "main_mo/envs/tree_env.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

#include <mo/mo_helper.h>

using namespace std; 
namespace py = pybind11;


namespace thts {
    ToyTreeEnv::ToyTreeEnv(int num_rewards, int num_actions, int tree_depth, bool sparse) : 
        MoThtsEnv(num_rewards, true),
        num_actions(num_actions),
        tree_depth(tree_depth),
        sparse(sparse),
        reward_vectors(thts::helper::get_well_spaced_hyperphere_points(num_actions,num_rewards))
    {
    }

    ToyTreeEnv::ToyTreeEnv(const ToyTreeEnv& other) :
        MoThtsEnv(other.reward_dim, true),
        num_actions(other.num_actions),
        tree_depth(other.tree_depth),
        sparse(other.sparse),
        reward_vectors(other.reward_vectors)
    {
    }

    shared_ptr<ThtsEnv> ToyTreeEnv::clone() {
        return std::dynamic_pointer_cast<ThtsEnv>(std::make_shared<ToyTreeEnv>(*this));
    }

    /**
     * Returns a vector full of zeros
     */
    shared_ptr<const IntVectorState> ToyTreeEnv::get_initial_state() const 
    {
        vector<int> init_state_list(num_actions+1, 0);
        return make_shared<const IntVectorState>(init_state_list);
    }

    bool ToyTreeEnv::is_sink_state(shared_ptr<const IntVectorState> state) const 
    {
        return state->state[0] == tree_depth;
    }

    shared_ptr<IntActionVector> ToyTreeEnv::get_valid_actions(shared_ptr<const IntVectorState> state) const 
    {
        shared_ptr<IntActionVector> valid_actions = make_shared<IntActionVector>();
        valid_actions->reserve(num_actions);
        for (int i=0; i<num_actions; i++) {
            valid_actions->push_back(make_shared<IntAction>(i));
        }
        return valid_actions;
    }

    shared_ptr<IntVectorState> ToyTreeEnv::get_next_state(
        shared_ptr<const IntVectorState> state, shared_ptr<const IntAction> action) const
    {
        vector<int> next_state_list(state->state);
        next_state_list[0]++;
        next_state_list[action->action+1]++;
        return make_shared<IntVectorState>(next_state_list);   
    }

    shared_ptr<IntVectorStateDistr> ToyTreeEnv::get_transition_distribution(
        shared_ptr<const IntVectorState> state, shared_ptr<const IntAction> action) const 
    {
        shared_ptr<IntVectorStateDistr> distr;
        shared_ptr<IntVectorState> next_state = get_next_state(state,action);
        distr->insert_or_assign(next_state,1.0);
        return distr;
    }

    shared_ptr<const IntVectorState> ToyTreeEnv::sample_transition_distribution(
        shared_ptr<const IntVectorState> state, shared_ptr<const IntAction> action, RandManager& rand_manager) const 
    {
        return get_next_state(state,action);
    }
 
    Eigen::ArrayXd ToyTreeEnv::get_mo_reward(
        shared_ptr<const IntVectorState> state, 
        shared_ptr<const IntAction> action) const 
    {
        if (!sparse) {
            return reward_vectors[action->action] / tree_depth;
        }

        int depth = state->state[0];
        if (depth != tree_depth-1) {
            return Eigen::ArrayXd::Zero(reward_dim);
        }

        Eigen::ArrayXd final_reward = Eigen::ArrayXd::Zero(reward_dim);
        for (int i=0; i<num_actions; i++) {
            final_reward += reward_vectors[i] * state->state[i+1] / tree_depth;
        }
        return final_reward;
    }
}



/**
 * Boilerplate ThtsEnv interface implementation. Copied from thts_env_template.h.
 * All this code basically calls the corresponding implementation function, with approprtiate casts before/after.
 */
namespace thts {
    
    shared_ptr<const State> ToyTreeEnv::get_initial_state_itfc() const 
    {
        shared_ptr<const IntVectorState> init_state = get_initial_state();
        return static_pointer_cast<const State>(init_state);
    }

    bool ToyTreeEnv::is_sink_state_itfc(shared_ptr<const State> state, ThtsContext& ctx) const 
    {
        shared_ptr<const IntVectorState> state_itfc = static_pointer_cast<const IntVectorState>(state);
        return is_sink_state(state_itfc);
    }

    shared_ptr<ActionVector> ToyTreeEnv::get_valid_actions_itfc(
        shared_ptr<const State> state, ThtsContext& ctx) const 
    {
        shared_ptr<const IntVectorState> state_itfc = static_pointer_cast<const IntVectorState>(state);
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
        shared_ptr<const IntVectorState> state_itfc = static_pointer_cast<const IntVectorState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<IntVectorStateDistr> distr_itfc = get_transition_distribution(state_itfc, action_itfc);
        
        shared_ptr<StateDistr> distr = make_shared<StateDistr>(); 
        for (pair<shared_ptr<const IntVectorState>,double> key_val_pair : *distr_itfc) {
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
        shared_ptr<const IntVectorState> state_itfc = static_pointer_cast<const IntVectorState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<const IntVectorState> obsv = sample_transition_distribution(state_itfc, action_itfc, rand_manager);
        return static_pointer_cast<const State>(obsv);
    }

    Eigen::ArrayXd ToyTreeEnv::get_mo_reward_itfc(
        shared_ptr<const State> state, 
        shared_ptr<const Action> action,
        ThtsContext& ctx) const
    {
        shared_ptr<const IntVectorState> state_itfc = static_pointer_cast<const IntVectorState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        return get_mo_reward(state_itfc, action_itfc);
    }
}