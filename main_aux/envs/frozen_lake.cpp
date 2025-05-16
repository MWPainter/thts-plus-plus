#include "main_aux/envs/frozen_lake.h"

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
     * Helper function - modulo, without the rubbish negative cases
     */
    inline int mod(const int a, const int b) {
        int res = a % b;
        return res < 0 ? res + b : res;
    }

    /**
     * Helper function - checking for goal state
     */
    bool is_goal_state(shared_ptr<const Int3TupleState> state, const std::string* map) {
        const int x = get<0>(state->state);
        const int y = get<1>(state->state);
        char location_char = map[x][y];
        return location_char == GOAL_CHAR;
    }

    /**
     * Helper function - checking for hole state
     */
    bool is_hole_state(shared_ptr<const Int3TupleState> state, const std::string* map) {
        const int x = get<0>(state->state);
        const int y = get<1>(state->state);
        char location_char = map[x][y];
        return location_char == HOLE_CHAR;
    }

    /**
     * Compute next location helper
     *  
     * Returns the values of x,y (passed by ref) of taking 'action' from 'state'
    */
    shared_ptr<const Int3TupleState> compute_next_state_deterministic(
        shared_ptr<const Int3TupleState> state, 
        const int act_val, 
        const int width, 
        const int height)  
    {
        int x = get<0>(state->state) + DELTA_X[act_val];
        int y = get<1>(state->state) + DELTA_Y[act_val];
        int t = get<2>(state->state);
        if (x < 0) x = 0;
        else if (x >= width) x = width-1;
        if (y < 0) y = 0;
        else if (y >= height) y = height-1;
        return make_shared<const Int3TupleState>(x, y, t+1);
    }

    shared_ptr<const Int3TupleState> compute_next_state_deterministic(
        shared_ptr<const Int3TupleState> state, 
        shared_ptr<const IntAction> action, 
        int width, 
        int height)  
    {
        return compute_next_state_deterministic(state, action->action, width, height);
    }

    /**
     * Constructor
    */
    FrozenLakeEnv::FrozenLakeEnv(
        int width, 
        int height, 
        const std::string* map,
        bool is_slippery, 
        int reward_type, 
        double reward_discount_factor, 
        int max_steps) : 
            ThtsEnv(true), 
            height(height),
            width(width),
            map(map),
            cached_actions(make_shared<IntActionVector>()),
            reward_type(reward_type),
            reward_discount_factor(reward_discount_factor),
            max_steps(max_steps == -1 ? 3*(width+height) : max_steps),
            is_slippery(is_slippery)
    {
        cached_actions->push_back(make_shared<const IntAction>(FL_RIGHT));
        cached_actions->push_back(make_shared<const IntAction>(FL_DOWN));
        cached_actions->push_back(make_shared<const IntAction>(FL_LEFT));
        cached_actions->push_back(make_shared<const IntAction>(FL_UP));
    }

    shared_ptr<ThtsEnv> FrozenLakeEnv::clone() {
        return make_shared<FrozenLakeEnv>(width,height,map,is_slippery,reward_type,reward_discount_factor,max_steps);
    }

    /**
     * Initial state = 0,0
    */
    shared_ptr<const Int3TupleState> FrozenLakeEnv::get_initial_state() const {
        return make_shared<Int3TupleState>(0,0,0);
    }

    /**
     * Cant move when fall in hole or at goal
     * 
     * To handle is_slippery case, after falling in a hole, we need the agent to make one more step so it can collect 
     * its "reward" for falling in the hole.
     * 
     * Also need to do same thing for goal state, so that we can collect the reward for reaching the goal.
    */
    bool FrozenLakeEnv::is_sink_state(shared_ptr<const Int3TupleState> state) const {
        const int t = get<2>(state->state);
        return (is_goal_state(state,map) && t==-1) || (is_hole_state(state,map) && t==-1) || t >= max_steps;
    }

    /**
     * Can always move in any direction, unless in sink state
     */    
    shared_ptr<IntActionVector> FrozenLakeEnv::get_valid_actions(shared_ptr<const Int3TupleState> state) const {
        return cached_actions;
    }

    /**
     * If in hole, then make one step where location doesnt move but set time to -1 to let know we've collected hole reward
    */
    shared_ptr<Int3TupleStateDistr> FrozenLakeEnv::get_transition_distribution(
        shared_ptr<const Int3TupleState> state, shared_ptr<const IntAction> action) const 
    {
        shared_ptr<Int3TupleStateDistr> next_state_distr = make_shared<Int3TupleStateDistr>();

        if (is_hole_state(state,map) || is_goal_state(state,map)) {
            shared_ptr<Int3TupleStateDistr> next_state_distr = make_shared<Int3TupleStateDistr>();
            shared_ptr<const Int3TupleState> next_state = make_shared<const Int3TupleState>(get<0>(state->state), get<1>(state->state), -1);
            next_state_distr->insert_or_assign(next_state, 1.0);
            return next_state_distr;
        }

        if (!is_slippery) {
            shared_ptr<const Int3TupleState> next_state = compute_next_state_deterministic(state, action, width, height);
            next_state_distr->insert_or_assign(next_state, 1.0);
            return next_state_distr;
        }

        double slip_prob = 1.0/3.0;
        double next_state_prob = slip_prob;

        shared_ptr<const Int3TupleState> next_state = compute_next_state_deterministic(state, action, width, height);
        next_state_distr->insert_or_assign(next_state, next_state_prob);

        next_state = compute_next_state_deterministic(state, mod(action->action-1,4), width, height);
        next_state_prob = slip_prob;
        if (next_state_distr->contains(next_state)) {
            next_state_prob += next_state_distr->at(next_state);
        }
        next_state_distr->insert_or_assign(next_state, next_state_prob);

        next_state_prob = 1.0 - 2*slip_prob;
        next_state = compute_next_state_deterministic(state, mod(action->action+1,4), width, height);
        if (next_state_distr->contains(next_state)) {
            next_state_prob += next_state_distr->at(next_state);
        }
        next_state_distr->insert_or_assign(next_state, next_state_prob);
        
        return next_state_distr;
    }

    /**
     * Deterministic env, call other version
    */
    shared_ptr<const Int3TupleState> FrozenLakeEnv::sample_transition_distribution(
        shared_ptr<const Int3TupleState> state, shared_ptr<const IntAction> action, RandManager& rand_manager) const 
    {
        if (is_hole_state(state,map) || is_goal_state(state,map)) {
            return make_shared<const Int3TupleState>(get<0>(state->state), get<1>(state->state), -1);
        }
        if (!is_slippery) {
            return compute_next_state_deterministic(state, action, width, height);
        }
        int rand_int = rand_manager.get_rand_int(0,3) - 1;
        return compute_next_state_deterministic(state, mod(action->action+rand_int,4), width, height);
    }

    /**
     * If going to reach goal state, then return reward of 1.0
     * 
     * First handle dense reward case, as its just return -1, apart from hole state
     * Then handle the sparse reward cases seperately
     * 
    */
    double FrozenLakeEnv::get_reward(
        shared_ptr<const Int3TupleState> state, 
        shared_ptr<const IntAction> action) const 
    {
        const int t = get<2>(state->state);

        if (reward_type == FL_DENSE_REWARD) {
            if (is_hole_state(state,map)) {
                return -(max_steps - t); 
            }
            return -1.0;
        }

        if (reward_type == FL_SPARSE_LEN_REWARD) {
            if (is_hole_state(state,map)) {
                return -max_steps;
            }
            if (is_goal_state(state,map)) {
                return -t;
            }
            if (t == max_steps-1) {
                return -max_steps;
            }
            return 0.0;
        }

        if (reward_type == FL_SPARSE_DISCOUNTED_REWARD) {
            if (is_goal_state(state,map)) {
                return pow(reward_discount_factor, t);
            }
            return 0.0;
        }

        string error_msg = "Invalid reward type used in frozen lake";
        throw runtime_error(error_msg);
    }
}



/**
 * Boilerplate defined functions. Copied from thts_env_template.h.
 */
namespace thts {
    shared_ptr<Int3TupleStateDistr> FrozenLakeEnv::get_observation_distribution(
        shared_ptr<const IntAction> action, shared_ptr<const Int3TupleState> next_state, ThtsEnvContext& ctx) const 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<ObservationDistr> distr_itfc = ThtsEnv::get_observation_distribution_itfc(
            act_itfc, next_state_itfc, ctx);
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
        RandManager& rand_manager, ThtsEnvContext& ctx) const 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<const Observation> obsv_itfc = ThtsEnv::sample_observation_distribution_itfc(
            act_itfc, next_state_itfc, rand_manager, ctx);
        return static_pointer_cast<const Int3TupleState>(obsv_itfc);
    }

    shared_ptr<ThtsEnvContext> FrozenLakeEnv::sample_context(int tid, RandManager& rand_manager) const
    {
        shared_ptr<ThtsEnvContext> context = ThtsEnv::sample_context_itfc(tid,rand_manager);
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

    bool FrozenLakeEnv::is_sink_state_itfc(shared_ptr<const State> state, ThtsEnvContext& ctx) const {
        shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
        return is_sink_state(state_itfc);
    }

    shared_ptr<ActionVector> FrozenLakeEnv::get_valid_actions_itfc(shared_ptr<const State> state, ThtsEnvContext& ctx) const {
        shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
        shared_ptr<vector<shared_ptr<const IntAction>>> valid_actions_itfc = get_valid_actions(state_itfc);

        shared_ptr<ActionVector> valid_actions = make_shared<ActionVector>();
        for (shared_ptr<const IntAction> act : *valid_actions_itfc) {
            valid_actions->push_back(static_pointer_cast<const Action>(act));
        }
        return valid_actions;
    }

    shared_ptr<StateDistr> FrozenLakeEnv::get_transition_distribution_itfc(
        shared_ptr<const State> state, shared_ptr<const Action> action, ThtsEnvContext& ctx) const 
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
       shared_ptr<const State> state, shared_ptr<const Action> action, RandManager& rand_manager, ThtsEnvContext& ctx) const 
    {
        shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<const Int3TupleState> obsv = sample_transition_distribution(state_itfc, action_itfc, rand_manager);
        return static_pointer_cast<const State>(obsv);
    }

    shared_ptr<ObservationDistr> FrozenLakeEnv::get_observation_distribution_itfc(
        shared_ptr<const Action> action, shared_ptr<const State> next_state, ThtsEnvContext& ctx) const
    {
        shared_ptr<const IntAction> act_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<const Int3TupleState> next_state_itfc = static_pointer_cast<const Int3TupleState>(next_state);
        shared_ptr<Int3TupleStateDistr> distr_itfc = get_observation_distribution(
            act_itfc, next_state_itfc, ctx);
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
         RandManager& rand_manager, ThtsEnvContext& ctx) const
    {
        shared_ptr<const IntAction> act_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<const Int3TupleState> next_state_itfc = static_pointer_cast<const Int3TupleState>(next_state);
        shared_ptr<const Int3TupleState> obsv_itfc = sample_observation_distribution(
            act_itfc, next_state_itfc, rand_manager, ctx);
        return static_pointer_cast<const Observation>(obsv_itfc);
    }

    double FrozenLakeEnv::get_reward_itfc(
        shared_ptr<const State> state, 
        shared_ptr<const Action> action, 
        ThtsEnvContext& ctx) const
    {
        shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        return get_reward(state_itfc, action_itfc);
    }

    shared_ptr<ThtsEnvContext> FrozenLakeEnv::sample_context_itfc(int tid, RandManager& rand_manager) const
    {
        shared_ptr<ThtsEnvContext> context = ThtsEnv::sample_context_itfc(tid, rand_manager);
        return static_pointer_cast<ThtsEnvContext>(context);
    }
}