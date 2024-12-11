#include "toy_envs/frozen_lake_env.h"

#include <array>
#include <cmath>

using namespace std; 

// enum FLDirection        { FL_RIGHT, FL_DOWN, FL_LEFT, FL_UP };
static const int DELTA_X[] = {      1,       0,      -1,     0 };
static const int DELTA_Y[] = {      0,       1,       0,    -1 };

static const char HOLE_CHAR = 'H';
static const char GOAL_CHAR = 'G';

namespace thts {
    /**
     * Constructor
    */
    FrozenLakeEnv::FrozenLakeEnv(int width, int height, const std::string* map, double reward_discount_factor) : 
        ThtsEnv(true), 
        height(height),
        width(width),
        map(map),
        cached_actions(make_shared<IntActionVector>()),
        reward_discount_factor(reward_discount_factor)
    {
        cached_actions->push_back(make_shared<const IntAction>(FL_RIGHT));
        cached_actions->push_back(make_shared<const IntAction>(FL_DOWN));
        cached_actions->push_back(make_shared<const IntAction>(FL_LEFT));
        cached_actions->push_back(make_shared<const IntAction>(FL_UP));
    }

    /**
     * Initial state = 0,0
    */
    shared_ptr<const Int3TupleState> FrozenLakeEnv::get_initial_state() const {
        return make_shared<Int3TupleState>(0,0,0);
    }

    /**
     * Cant move when fall in hole or at goal
    */
    bool FrozenLakeEnv::is_sink_state(shared_ptr<const Int3TupleState> state) const {
        const int x = get<0>(state->state);
        const int y = get<1>(state->state);
        char location_char = map[x][y];
        return location_char == HOLE_CHAR || location_char == GOAL_CHAR;
    }

    /**
     * Can always move in any direction, unless in sink state
     */    
    shared_ptr<IntActionVector> FrozenLakeEnv::get_valid_actions(shared_ptr<const Int3TupleState> state) const {
        if (is_sink_state(state)) return make_shared<IntActionVector>();
        return cached_actions;
    }

    /**
     * Distribution = next state with prob 1
    */
    shared_ptr<Int3TupleStateDistr> FrozenLakeEnv::get_transition_distribution(
        shared_ptr<const Int3TupleState> state, shared_ptr<const IntAction> action) const 
    {
        shared_ptr<const Int3TupleState> next_state = sample_transition_distribution(state, action);
        shared_ptr<Int3TupleStateDistr> next_state_distr = make_shared<Int3TupleStateDistr>();
        next_state_distr->insert_or_assign(next_state, 1.0);
        return next_state_distr;
    }

    /**
     * Compute next location helper
     * 
     * Returns the values of x,y (passed by ref) of taking 'action' from 'state'
    */
    void compute_next_loc(
        int& x, 
        int& y, 
        shared_ptr<const Int3TupleState> state, 
        shared_ptr<const IntAction> action, 
        int width, 
        int height)  
    {
        int act_val = action->action;
        x = get<0>(state->state) + DELTA_X[act_val];
        y = get<1>(state->state) + DELTA_Y[act_val];
        if (x < 0) x = 0;
        else if (x >= width) x = width-1;
        if (y < 0) y = 0;
        else if (y >= height) y = height-1;
    }

    /**
     * Compute next state
    */
    shared_ptr<const Int3TupleState> FrozenLakeEnv::sample_transition_distribution(
        shared_ptr<const Int3TupleState> state, shared_ptr<const IntAction> action) const 
    {
        int x,y;
        compute_next_loc(x,y, state, action, width, height);
        const int t = get<2>(state->state);
        shared_ptr<const Int3TupleState> next_state = make_shared<const Int3TupleState>(x,y,t+1);
        return next_state;
    }

    /**
     * Deterministic env, call other version
    */
    shared_ptr<const Int3TupleState> FrozenLakeEnv::sample_transition_distribution(
        shared_ptr<const Int3TupleState> state, shared_ptr<const IntAction> action, RandManager& rand_manager) const 
    {
        return sample_transition_distribution(state,action);
    }

    /**
     * If going to reach goal state, then return reward of 1.0
    */
    double FrozenLakeEnv::get_reward(
        shared_ptr<const Int3TupleState> state, 
        shared_ptr<const IntAction> action, 
        shared_ptr<const Int3TupleState> observation) const 
    {
        int x,y;
        compute_next_loc(x,y, state, action, width, height);
        char next_location_char = map[x][y];
        if (next_location_char == GOAL_CHAR) {
            const int t = get<2>(state->state);
            return pow(reward_discount_factor, t+1);
        }
        return 0.0;
    }
}



/**
 * Boilerplate defined functions. Copied from thts_env_template.h.
 */
namespace thts {
    shared_ptr<Int3TupleStateDistr> FrozenLakeEnv::get_observation_distribution(
        shared_ptr<const IntAction> action, shared_ptr<const Int3TupleState> next_state) const 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<ObservationDistr> distr_itfc = ThtsEnv::get_observation_distribution_itfc(
            act_itfc, next_state_itfc);
        shared_ptr<Int3TupleStateDistr> distr;
        for (pair<const shared_ptr<const Observation>,double> pr : *distr_itfc) {
            shared_ptr<const Int3TupleState> obsv = static_pointer_cast<const Int3TupleState>(pr.first);
            distr->insert_or_assign(obsv, pr.second);
        }
        return distr;
    }

    shared_ptr<const Int3TupleState> FrozenLakeEnv::sample_observation_distribution(
        shared_ptr<const IntAction> action, 
        shared_ptr<const Int3TupleState> next_state, 
        RandManager& rand_manager) const 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<const Observation> obsv_itfc = ThtsEnv::sample_observation_distribution_itfc(
            act_itfc, next_state_itfc, rand_manager);
        return static_pointer_cast<const Int3TupleState>(obsv_itfc);
    }

    shared_ptr<ThtsEnvContext> FrozenLakeEnv::sample_context(shared_ptr<const Int3TupleState> state) const
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
    
    shared_ptr<const State> FrozenLakeEnv::get_initial_state_itfc() const {
        shared_ptr<const Int3TupleState> init_state = get_initial_state();
        return static_pointer_cast<const State>(init_state);
    }

    bool FrozenLakeEnv::is_sink_state_itfc(shared_ptr<const State> state) const {
        shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
        return is_sink_state(state_itfc);
    }

    shared_ptr<ActionVector> FrozenLakeEnv::get_valid_actions_itfc(shared_ptr<const State> state) const {
        shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
        shared_ptr<vector<shared_ptr<const IntAction>>> valid_actions_itfc = get_valid_actions(state_itfc);

        shared_ptr<ActionVector> valid_actions = make_shared<ActionVector>();
        for (shared_ptr<const IntAction> act : *valid_actions_itfc) {
            valid_actions->push_back(static_pointer_cast<const Action>(act));
        }
        return valid_actions;
    }

    shared_ptr<StateDistr> FrozenLakeEnv::get_transition_distribution_itfc(
        shared_ptr<const State> state, shared_ptr<const Action> action) const 
    {
        shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<Int3TupleStateDistr> distr_itfc = get_transition_distribution(state_itfc, action_itfc);
        
        shared_ptr<StateDistr> distr = make_shared<StateDistr>(); 
        for (pair<shared_ptr<const Int3TupleState>,double> key_val_pair : *distr_itfc) {
            shared_ptr<const State> obsv = static_pointer_cast<const State>(key_val_pair.first);
            double prob = key_val_pair.second;
            distr->insert_or_assign(obsv, prob);
        }
        return distr;
    }

    shared_ptr<const State> FrozenLakeEnv::sample_transition_distribution_itfc(
       shared_ptr<const State> state, shared_ptr<const Action> action, RandManager& rand_manager) const 
    {
        shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<const Int3TupleState> obsv = sample_transition_distribution(state_itfc, action_itfc, rand_manager);
        return static_pointer_cast<const State>(obsv);
    }

    shared_ptr<ObservationDistr> FrozenLakeEnv::get_observation_distribution_itfc(
        shared_ptr<const Action> action, shared_ptr<const State> next_state) const
    {
        shared_ptr<const IntAction> act_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<const Int3TupleState> next_state_itfc = static_pointer_cast<const Int3TupleState>(next_state);
        shared_ptr<Int3TupleStateDistr> distr_itfc = get_observation_distribution(
            act_itfc, next_state_itfc);
        shared_ptr<ObservationDistr> distr;
        for (pair<const shared_ptr<const Int3TupleState>,double> pr : *distr_itfc) {
            shared_ptr<const Observation> obsv = static_pointer_cast<const Observation>(pr.first);
            distr->insert_or_assign(obsv, pr.second);
        }
        return distr;
    }     

    shared_ptr<const Observation> FrozenLakeEnv::sample_observation_distribution_itfc(
        shared_ptr<const Action> action, 
        shared_ptr<const State> next_state,
         RandManager& rand_manager) const
    {
        shared_ptr<const IntAction> act_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<const Int3TupleState> next_state_itfc = static_pointer_cast<const Int3TupleState>(next_state);
        shared_ptr<const Int3TupleState> obsv_itfc = sample_observation_distribution(
            act_itfc, next_state_itfc, rand_manager);
        return static_pointer_cast<const Observation>(obsv_itfc);
    }

    double FrozenLakeEnv::get_reward_itfc(
        shared_ptr<const State> state, 
        shared_ptr<const Action> action, 
        shared_ptr<const Observation> observation) const
    {
        shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<const Int3TupleState> obsv_itfc = static_pointer_cast<const Int3TupleState>(observation);
        return get_reward(state_itfc, action_itfc, obsv_itfc);
    }

    shared_ptr<ThtsEnvContext> FrozenLakeEnv::sample_context_itfc(shared_ptr<const State> state) const
    {
        shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
        shared_ptr<ThtsEnvContext> context = sample_context(state_itfc);
        return static_pointer_cast<ThtsEnvContext>(context);
    }
}