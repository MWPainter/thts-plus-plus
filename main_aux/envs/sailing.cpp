#include "main_aux/envs/sailing.h"

#include "helper_templates.h"

#include <cmath>
#include <sstream>

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

    size_t SailingState::hash() const {
        size_t h = 0;
        h = helper::hash_combine(h, x);
        h = helper::hash_combine(h, y);
        h = helper::hash_combine(h, wind_dir);
        return helper::hash_combine(h, timestep);
    }

    bool SailingState::equals(const SailingState& other) const {
        return x == other.x && y == other.y && wind_dir == other.wind_dir && timestep == other.timestep;
    }

    bool SailingState::equals_itfc(const Observation& other) const {
        try {
            const SailingState& o = dynamic_cast<const SailingState&>(other);
            return equals(o);
        } catch (const bad_cast&) {
            return false;
        }
    }

    string SailingState::get_pretty_print_string() const {
        stringstream ss;
        ss << "(" << x << "," << y << "," << wind_dir << "," << timestep << ")";
        return ss.str();
    }

    ostream& operator<<(ostream& os, const SailingState& s) {
        os << s.get_pretty_print_string();
        return os;
    }

    ostream& operator<<(ostream& os, const shared_ptr<const SailingState>& s) {
        os << s->get_pretty_print_string();
        return os;
    }
    /**
     * Construct
    */
    SailingEnv::SailingEnv(int width, int height, int init_wind_dir) : 
        ThtsEnv(true), width(width), height(height), init_wind_dir(init_wind_dir)
    {
    }

    shared_ptr<ThtsEnv> SailingEnv::clone() {
        return make_shared<SailingEnv>(width,height,init_wind_dir);
    }

    /**
     * Initial state is at 0,0 with wind init_wind_dir and timestep 0
    */
    shared_ptr<const SailingState> SailingEnv::get_initial_state() const {
        return make_shared<SailingState>(0, 0, init_wind_dir, 0);
    }

    /**
     * Only sink is by being at goal at other side of grid
    */
    bool SailingEnv::is_sink_state(shared_ptr<const SailingState> state) const {
        return (state->x == width - 1 && state->y == height - 1);
    }

    /**
     * If at goal, no actions
     * Don't allow sailing off edge
     * Don't allow sailing directly against wind
    */
    shared_ptr<IntActionVector> SailingEnv::get_valid_actions(shared_ptr<const SailingState> state) const {
        shared_ptr<IntActionVector> valid_actions = make_shared<IntActionVector>();
        if (is_sink_state(state)) return valid_actions;

        int x = state->x;
        int y = state->y;
        int w = state->wind_dir;

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
        } else if (y == height-1) {
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
     * Get the current state (x,y,wind_dir,timestep)
     * Gets the delta for sailing direction (action) of dx, dy
     * Iterates through all possible new wind directions nw and adds (x+dx, y+dy, nw, timestep+1) if the transition
     * probability of w -> nw is positive
    */
    shared_ptr<SailingStateDistr> SailingEnv::get_transition_distribution(
        shared_ptr<const SailingState> state, shared_ptr<const IntAction> action) const
    {
        int x = state->x;
        int y = state->y;
        int w = state->wind_dir;
        int next_timestep = state->timestep + 1;
        int delta_x = DELTA_X[action->action];
        int delta_y = DELTA_Y[action->action];

        shared_ptr<SailingStateDistr> distr = make_shared<SailingStateDistr>();
        for (int new_w = 0; new_w < 8; new_w++) {
            double prob = WIND_TRANSITION_PROBS[w][new_w];
            if (prob > 0) {
                shared_ptr<SailingState> next_state = make_shared<SailingState>(
                    x + delta_x, y + delta_y, new_w, next_timestep);
                distr->insert_or_assign(next_state, prob);
            }
        }

        return distr;
    }

    /**
     * Samples from distribution constructed in get_transition_distribution
    */
    shared_ptr<const SailingState> SailingEnv::sample_transition_distribution(
        shared_ptr<const SailingState> state, shared_ptr<const IntAction> action, RandManager& rand_manager) const
    {
        shared_ptr<SailingStateDistr> distr = get_transition_distribution(state, action);
        return helper::sample_from_distribution(*distr, rand_manager);
    }

    /**
     * tack = direction relative to wind
     * intuitively travelling with the wind costs less
     * cost of travelling = -1.0 - tack, where tack is the number of 45deg turns from the direction of wind
    */
    double SailingEnv::get_reward(
        shared_ptr<const SailingState> state,
        shared_ptr<const IntAction> action) const
    {
        int w = state->wind_dir;
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
    shared_ptr<SailingStateDistr> SailingEnv::get_observation_distribution(
        shared_ptr<const IntAction> action, shared_ptr<const SailingState> next_state, ThtsContext& ctx) const
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<ObservationDistr> distr_itfc = ThtsEnv::get_observation_distribution_itfc(
            act_itfc, next_state_itfc, ctx);
        shared_ptr<SailingStateDistr> distr = make_shared<SailingStateDistr>();
        for (pair<const shared_ptr<const Observation>, double> pr : *distr_itfc) {
            shared_ptr<const SailingState> obsv = static_pointer_cast<const SailingState>(pr.first);
            distr->insert_or_assign(obsv, pr.second);
        }
        return distr;
    }

    shared_ptr<const SailingState> SailingEnv::sample_observation_distribution(
        shared_ptr<const IntAction> action,
        shared_ptr<const SailingState> next_state,
        RandManager& rand_manager, ThtsContext& ctx) const
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<const Observation> obsv_itfc = ThtsEnv::sample_observation_distribution_itfc(
            act_itfc, next_state_itfc, rand_manager, ctx);
        return static_pointer_cast<const SailingState>(obsv_itfc);
    }

    shared_ptr<ThtsContext> SailingEnv::sample_context(int tid, RandManager& rand_manager) const
    {
        shared_ptr<ThtsContext> context = ThtsEnv::sample_context_itfc(tid,rand_manager);
        return static_pointer_cast<ThtsContext>(context);
    }
}



/**
 * Boilerplate ThtsEnv interface implementation. Copied from thts_env_template.h.
 * All this code basically calls the corresponding implementation function, with approprtiate casts before/after.
 */
namespace thts {

    shared_ptr<const State> SailingEnv::get_initial_state_itfc() const {
        shared_ptr<const SailingState> init_state = get_initial_state();
        return static_pointer_cast<const State>(init_state);
    }

    bool SailingEnv::is_sink_state_itfc(shared_ptr<const State> state, ThtsContext& ctx) const {
        shared_ptr<const SailingState> state_itfc = static_pointer_cast<const SailingState>(state);
        return is_sink_state(state_itfc);
    }

    shared_ptr<ActionVector> SailingEnv::get_valid_actions_itfc(shared_ptr<const State> state, ThtsContext& ctx) const {
        shared_ptr<const SailingState> state_itfc = static_pointer_cast<const SailingState>(state);
        shared_ptr<vector<shared_ptr<const IntAction>>> valid_actions_itfc = get_valid_actions(state_itfc);

        shared_ptr<ActionVector> valid_actions = make_shared<ActionVector>();
        for (shared_ptr<const IntAction> act : *valid_actions_itfc) {
            valid_actions->push_back(static_pointer_cast<const Action>(act));
        }
        return valid_actions;
    }

    shared_ptr<StateDistr> SailingEnv::get_transition_distribution_itfc(
        shared_ptr<const State> state, shared_ptr<const Action> action, ThtsContext& ctx) const
    {
        shared_ptr<const SailingState> state_itfc = static_pointer_cast<const SailingState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<SailingStateDistr> distr_itfc = get_transition_distribution(state_itfc, action_itfc);

        shared_ptr<StateDistr> distr = make_shared<StateDistr>();
        for (pair<shared_ptr<const SailingState>, double> key_val_pair : *distr_itfc) {
            shared_ptr<const State> obsv = static_pointer_cast<const State>(key_val_pair.first);
            double prob = key_val_pair.second;
            distr->insert_or_assign(obsv, prob);
        }
        return distr;
    }

    shared_ptr<const State> SailingEnv::sample_transition_distribution_itfc(
        shared_ptr<const State> state, shared_ptr<const Action> action, RandManager& rand_manager, ThtsContext& ctx) const
    {
        shared_ptr<const SailingState> state_itfc = static_pointer_cast<const SailingState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<const SailingState> obsv = sample_transition_distribution(state_itfc, action_itfc, rand_manager);
        return static_pointer_cast<const State>(obsv);
    }

    shared_ptr<ObservationDistr> SailingEnv::get_observation_distribution_itfc(
        shared_ptr<const Action> action, shared_ptr<const State> next_state, ThtsContext& ctx) const
    {
        shared_ptr<const IntAction> act_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<const SailingState> next_state_itfc = static_pointer_cast<const SailingState>(next_state);
        shared_ptr<SailingStateDistr> distr_itfc = get_observation_distribution(
            act_itfc, next_state_itfc, ctx);
        shared_ptr<ObservationDistr> distr = make_shared<ObservationDistr>();
        for (pair<const shared_ptr<const SailingState>, double> pr : *distr_itfc) {
            shared_ptr<const Observation> obsv = static_pointer_cast<const Observation>(pr.first);
            distr->insert_or_assign(obsv, pr.second);
        }
        return distr;
    }

    shared_ptr<const Observation> SailingEnv::sample_observation_distribution_itfc(
        shared_ptr<const Action> action,
        shared_ptr<const State> next_state,
        RandManager& rand_manager, ThtsContext& ctx) const
    {
        shared_ptr<const IntAction> act_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<const SailingState> next_state_itfc = static_pointer_cast<const SailingState>(next_state);
        shared_ptr<const SailingState> obsv_itfc = sample_observation_distribution(
            act_itfc, next_state_itfc, rand_manager, ctx);
        return static_pointer_cast<const Observation>(obsv_itfc);
    }

    double SailingEnv::get_reward_itfc(
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        ThtsContext& ctx) const
    {
        shared_ptr<const SailingState> state_itfc = static_pointer_cast<const SailingState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        return get_reward(state_itfc, action_itfc);
    }

    shared_ptr<ThtsContext> SailingEnv::sample_context_itfc(int tid, RandManager& rand_manager) const
    {
        shared_ptr<ThtsContext> context = sample_context(tid, rand_manager);
        return static_pointer_cast<ThtsContext>(context);
    }
}