#include "toy_envs/sailing_env.h"

#include "helper_templates.h"

#include <cmath>

// enum SailDirection       { NN, NE, EE, SE, SS, SW, WW, NW };
static const int DELTA_X[] = { 0,  1,  1,  1,  0, -1, -1, -1 };
static const int DELTA_Y[] = { 1,  1,  0, -1, -1, -1,  0,  1 };

static const double WIND_TRANSITION_PROBS[][8] {
    {0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3},
    {0.4, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.4, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.4, 0.3, 0.3, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.4, 0.2, 0.4, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.4, 0.0},
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.4},
    {0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3},
};

using namespace std; 


namespace thts {
    /**
     * Construct
    */
    SailingEnv::SailingEnv(int width, int height, int init_wind_dir) : 
        ThtsEnv(true), width(width), height(height), init_wind_dir(init_wind_dir)
    {
    }

    /**
     * Initial state is at 0,0 with wind NN direction
    */
    shared_ptr<const Int3TupleState> SailingEnv::get_initial_state() const {
        return make_shared<Int3TupleState>(0,0,init_wind_dir);
    }  

    /**
     * Only sink is by being at goal at other side of grid
    */
    bool SailingEnv::is_sink_state(shared_ptr<const Int3TupleState> state) const {
        int x = get<0>(state->state);
        int y = get<1>(state->state);
        return (x == width-1 && y == height-1);
    }

    /**
     * If at goal, no actions
     * Don't allow sailing off edge
     * Don't allow sailing directly against wind
    */
    shared_ptr<IntActionVector> SailingEnv::get_valid_actions(shared_ptr<const Int3TupleState> state) const {
        shared_ptr<IntActionVector> valid_actions = make_shared<IntActionVector>();
        if (is_sink_state(state)) return valid_actions;

        int x = get<0>(state->state);
        int y = get<1>(state->state);
        int w = get<2>(state->state);

        bool allowed_actions[8] = {true, true, true, true, true, true, true, true};

        if (x == 0) {
            allowed_actions[NW] = false;
            allowed_actions[WW] = false;
            allowed_actions[SW] = false;
        } else if (x == width-1) {
            allowed_actions[NE] = false;
            allowed_actions[EE] = false;
            allowed_actions[SE] = false;
        }

        if (y == 0) {
            allowed_actions[SE] = false;
            allowed_actions[SS] = false;
            allowed_actions[SW] = false;
        } else if (y == width-1) {
            allowed_actions[NE] = false;
            allowed_actions[NN] = false;
            allowed_actions[NW] = false;
        }

        int against_wind_dir = w-4;
        if (against_wind_dir < 0) against_wind_dir += 8;
        allowed_actions[against_wind_dir] = false;

        for (int i=0; i<8; i++) {
            if (allowed_actions[i]) {
                valid_actions->push_back(make_shared<IntAction>(i));
            }
        }

        return valid_actions;
    }

    /**
     * Assumes action won't move off grid
     * Get the current state (x,y,w)
     * Gets the delta for sailing direction (action) of dx, dy
     * Iterates through all possible new wind directions nw and adds (x+dx, y+dy, nw) if the transition probability of 
     * w -> nw is positive
    */
    shared_ptr<Int3TupleStateDistr> SailingEnv::get_transition_distribution(
        shared_ptr<const Int3TupleState> state, shared_ptr<const IntAction> action) const 
    {
        int x = get<0>(state->state);
        int y = get<1>(state->state);
        int w = get<2>(state->state);
        int delta_x = DELTA_X[action->action];
        int delta_y = DELTA_Y[action->action];

        shared_ptr<Int3TupleStateDistr> distr = make_shared<Int3TupleStateDistr>();
        for (int new_w=0; new_w<8; new_w++) {
            double prob = WIND_TRANSITION_PROBS[w][new_w];
            if (prob > 0) {
                shared_ptr<Int3TupleState> next_state = make_shared<Int3TupleState>(x+delta_x, y+delta_y, new_w);
                distr->insert_or_assign(next_state, prob);
            }
        }

        return distr;
    }

    /**
     * Samples from distribution constructed in get_transition_distribution
    */
    shared_ptr<const Int3TupleState> SailingEnv::sample_transition_distribution(
        shared_ptr<const Int3TupleState> state, shared_ptr<const IntAction> action, RandManager& rand_manager) const 
    {
        shared_ptr<Int3TupleStateDistr> distr = get_transition_distribution(state, action);
        return helper::sample_from_distribution(*distr, rand_manager);
    }
    
    /**
     * tack = direction relative to wind
     * intuitively travelling with the wind costs less
     * cost of travelling = -1.0 - tack, where tack is the number of 45deg turns from the direction of wind
    */
    double SailingEnv::get_reward(
        shared_ptr<const Int3TupleState> state, 
        shared_ptr<const IntAction> action, 
        shared_ptr<const Int3TupleState> observation) const 
    {
        int w = get<2>(state->state);
        double tack = abs(action->action-w);
        tack = fmin(tack, 8.0-tack);
        return -1.0 - tack;
    }
}



/**
 * Boilerplate defined functions. Copied from thts_env_template.h.
 * 
 * TODO: decide if need to write a custom version of these depending on if need partial observability or if need 
 * custom contexts.
 */
namespace thts {
    shared_ptr<Int3TupleStateDistr> SailingEnv::get_observation_distribution(
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

    shared_ptr<const Int3TupleState> SailingEnv::sample_observation_distribution(
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

    shared_ptr<ThtsEnvContext> SailingEnv::sample_context(shared_ptr<const Int3TupleState> state) const
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
    
    shared_ptr<const State> SailingEnv::get_initial_state_itfc() const {
        shared_ptr<const Int3TupleState> init_state = get_initial_state();
        return static_pointer_cast<const State>(init_state);
    }

    bool SailingEnv::is_sink_state_itfc(shared_ptr<const State> state) const {
        shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
        return is_sink_state(state_itfc);
    }

    shared_ptr<ActionVector> SailingEnv::get_valid_actions_itfc(shared_ptr<const State> state) const {
        shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
        shared_ptr<vector<shared_ptr<const IntAction>>> valid_actions_itfc = get_valid_actions(state_itfc);

        shared_ptr<ActionVector> valid_actions = make_shared<ActionVector>();
        for (shared_ptr<const IntAction> act : *valid_actions_itfc) {
            valid_actions->push_back(static_pointer_cast<const Action>(act));
        }
        return valid_actions;
    }

    shared_ptr<StateDistr> SailingEnv::get_transition_distribution_itfc(
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

    shared_ptr<const State> SailingEnv::sample_transition_distribution_itfc(
       shared_ptr<const State> state, shared_ptr<const Action> action, RandManager& rand_manager) const 
    {
        shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<const Int3TupleState> obsv = sample_transition_distribution(state_itfc, action_itfc, rand_manager);
        return static_pointer_cast<const State>(obsv);
    }

    shared_ptr<ObservationDistr> SailingEnv::get_observation_distribution_itfc(
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

    shared_ptr<const Observation> SailingEnv::sample_observation_distribution_itfc(
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

    double SailingEnv::get_reward_itfc(
        shared_ptr<const State> state, 
        shared_ptr<const Action> action, 
        shared_ptr<const Observation> observation) const
    {
        shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<const Int3TupleState> obsv_itfc = static_pointer_cast<const Int3TupleState>(observation);
        return get_reward(state_itfc, action_itfc, obsv_itfc);
    }

    shared_ptr<ThtsEnvContext> SailingEnv::sample_context_itfc(shared_ptr<const State> state) const
    {
        shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
        shared_ptr<ThtsEnvContext> context = sample_context(state_itfc);
        return static_pointer_cast<ThtsEnvContext>(context);
    }
}