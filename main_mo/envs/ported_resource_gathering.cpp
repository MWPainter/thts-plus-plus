#include "ported_resource_gathering.h"
#include <sstream>
#include "helper_templates.h"

using namespace std;

namespace thts {
    /**
     * ResourceGatheringState implementation
     */
    size_t ResourceGatheringState::hash() const {
        size_t h = 0;
        for (int i = 0; i < 5; i++) {
            h = thts::helper::hash_combine(h, state[i]);
        }
        return h;
    }

    bool ResourceGatheringState::equals(const ResourceGatheringState& other) const {
        return state == other.state;
    }

    bool ResourceGatheringState::equals_itfc(const Observation& other) const {
        const ResourceGatheringState* other_state = 
            dynamic_cast<const ResourceGatheringState*>(&other);
        if (other_state == nullptr) {
            return false;
        }
        return equals(*other_state);
    }

    string ResourceGatheringState::get_pretty_print_string() const {
        ostringstream oss;
        oss << "RGState(" << state[0] << "," << state[1] 
            << "," << state[2] << "," << state[3] << ",t=" << state[4] << ")";
        return oss.str();
    }

    /**
     * ResourceGatheringPreDeathState implementation
     */
    size_t ResourceGatheringPreDeathState::hash() const {
        size_t h = 0xDEADBEEF - 1;
        return thts::helper::hash_combine(h, timestep);
    }

    bool ResourceGatheringPreDeathState::equals(const ResourceGatheringPreDeathState& other) const {
        return timestep == other.timestep;
    }

    bool ResourceGatheringPreDeathState::equals_itfc(const Observation& other) const {
        const ResourceGatheringPreDeathState* other_state = 
            dynamic_cast<const ResourceGatheringPreDeathState*>(&other);
        if (other_state == nullptr) return false;
        return equals(*other_state);
    }

    string ResourceGatheringPreDeathState::get_pretty_print_string() const {
        ostringstream oss;
        oss << "RGPreDeathState(t=" << timestep << ")";
        return oss.str();
    }

    /**
     * ResourceGatheringTerminalState implementation
     */
    size_t ResourceGatheringTerminalState::hash() const {
        size_t h = 0xDEADBEEF;
        return thts::helper::hash_combine(h, timestep);
    }

    bool ResourceGatheringTerminalState::equals(const ResourceGatheringTerminalState& other) const {
        return timestep == other.timestep;
    }

    bool ResourceGatheringTerminalState::equals_itfc(const Observation& other) const {
        const ResourceGatheringTerminalState* other_state = 
            dynamic_cast<const ResourceGatheringTerminalState*>(&other);
        if (other_state == nullptr) return false;
        return equals(*other_state);
    }

    string ResourceGatheringTerminalState::get_pretty_print_string() const {
        ostringstream oss;
        oss << "RGTerminalState(t=" << timestep << ")";
        return oss.str();
    }

    /**
     * PortedResourceGatheringThtsEnv implementation
     */
    std::shared_ptr<ThtsEnv> PortedResourceGatheringThtsEnv::clone() {
        return std::dynamic_pointer_cast<ThtsEnv>(
            std::make_shared<PortedResourceGatheringThtsEnv>(*this));
    }

    char PortedResourceGatheringThtsEnv::get_cell_type(int x, int y) const {
        if (x == 0 && y == 2) return '1';  // R1
        if (x == 1 && y == 4) return '2';  // R2
        if (x == 0 && y == 3) return '2';   // E2
        if (x == 1 && y == 2) return '1';  // E1
        return ' ';
    }

    int PortedResourceGatheringThtsEnv::get_x(shared_ptr<const ResourceGatheringState> state) const {
        return state->state[0];
    }

    int PortedResourceGatheringThtsEnv::get_y(shared_ptr<const ResourceGatheringState> state) const {
        return state->state[1];
    }

    int PortedResourceGatheringThtsEnv::get_has_gold(shared_ptr<const ResourceGatheringState> state) const {
        return state->state[2];
    }

    int PortedResourceGatheringThtsEnv::get_has_gem(shared_ptr<const ResourceGatheringState> state) const {
        return state->state[3];
    }

    int PortedResourceGatheringThtsEnv::get_timestep(shared_ptr<const ResourceGatheringState> state) const {
        return state->state[4];
    }

    bool PortedResourceGatheringThtsEnv::is_valid_position(int x, int y) const {
        return x >= 0 && x < SIZE && y >= 0 && y < SIZE;
    }

    char PortedResourceGatheringThtsEnv::get_map_value(int x, int y) const {
        if (!is_valid_position(x, y)) return ' ';
        return MAP[x][y];
    }

    shared_ptr<const ResourceGatheringState> PortedResourceGatheringThtsEnv::get_initial_state() const {
        return make_shared<ResourceGatheringState>(INITIAL_X, INITIAL_Y, 0, 0, 0);
    }

    shared_ptr<const ResourceGatheringPreDeathState> PortedResourceGatheringThtsEnv::get_pre_death_state(int timestep) const {
        return make_shared<ResourceGatheringPreDeathState>(timestep);
    }

    shared_ptr<const ResourceGatheringTerminalState> PortedResourceGatheringThtsEnv::get_terminal_state(int timestep) const {
        return make_shared<ResourceGatheringTerminalState>(timestep);
    }

    bool PortedResourceGatheringThtsEnv::is_sink_state(shared_ptr<const State> state, ThtsContext& ctx) const {
        if (dynamic_cast<const ResourceGatheringTerminalState*>(state.get()) != nullptr) {
            return true;
        }
        const ResourceGatheringState* rg_state = 
            dynamic_cast<const ResourceGatheringState*>(state.get());
        if (rg_state != nullptr && rg_state->state[4] >= time_horizon) {
            return true;
        }
        return false;
    }

    shared_ptr<IntActionVector> PortedResourceGatheringThtsEnv::get_valid_actions(
        shared_ptr<const ResourceGatheringState> state, ThtsContext& ctx) const 
    {
        shared_ptr<IntActionVector> valid_actions = make_shared<IntActionVector>();
        shared_ptr<const State> state_ptr = static_pointer_cast<const State>(state);
        if (is_sink_state(state_ptr, ctx)) {
            return valid_actions;
        }
        for (int i = 0; i < 4; i++) {
            valid_actions->push_back(make_shared<const IntAction>(i));
        }
        return valid_actions;
    }

    vector<shared_ptr<const State>> PortedResourceGatheringThtsEnv::get_all_states() const {
        vector<shared_ptr<const State>> all_states_vec;
        
        // Enumerate states without timestep variation (timestep=0 representatives).
        // With graph search + timesteps, the full state space is unbounded,
        // but this gives the set of distinct spatial configurations.
        for (int t = 0; t <= time_horizon; t++) 
        {
            for (int x = 0; x < SIZE; x++) 
            {
                for (int y = 0; y < SIZE; y++) 
                {
                    for (int has_gold = 0; has_gold <= 1; has_gold++) 
                    {
                        for (int has_gem = 0; has_gem <= 1; has_gem++) 
                        {
                            all_states_vec.push_back(
                                static_pointer_cast<const State>(
                                    make_shared<ResourceGatheringState>(x, y, has_gold, has_gem, t)));
                        }
                    }
                }
            }
        }
        for (int t = 0; t <= time_horizon; t++)
        {
            all_states_vec.push_back(
                static_pointer_cast<const State>(get_pre_death_state(t)));
            all_states_vec.push_back(
                static_pointer_cast<const State>(get_terminal_state(t)));
        }
        
        return all_states_vec;
    }

    shared_ptr<const ResourceGatheringState> PortedResourceGatheringThtsEnv::make_next_state(
        shared_ptr<const ResourceGatheringState> state, 
        shared_ptr<const IntAction> action) const
    {
        int x = get_x(state);
        int y = get_y(state);
        int has_gold = get_has_gold(state);
        int has_gem = get_has_gem(state);
        int timestep = get_timestep(state);
        
        int next_x = x + DIR[action->action][0];
        int next_y = y + DIR[action->action][1];
        
        if (!is_valid_position(next_x, next_y)) {
            next_x = x;
            next_y = y;
        }
        
        char cell = get_map_value(next_x, next_y);
        char cell_type = get_cell_type(next_x, next_y);
        if (cell == 'R') {
            if (cell_type == '1') {
                has_gold = 1;
            } else if (cell_type == '2') {
                has_gem = 1;
            }
        }
        
        return make_shared<ResourceGatheringState>(next_x, next_y, has_gold, has_gem, timestep + 1);
    }

    shared_ptr<StateDistr> PortedResourceGatheringThtsEnv::get_transition_distribution(
        shared_ptr<const ResourceGatheringState> state, 
        shared_ptr<const IntAction> action, 
        ThtsContext& ctx) const 
    {
        shared_ptr<StateDistr> transition_distribution = make_shared<StateDistr>();
        
        shared_ptr<const ResourceGatheringState> next_state = make_next_state(state, action);
        int next_x = get_x(next_state);
        int next_y = get_y(next_state);
        int next_timestep = get_timestep(next_state);
        char cell = get_map_value(next_x, next_y);
        
        if (cell == 'H') {
            shared_ptr<const ResourceGatheringTerminalState> terminal_state = get_terminal_state(next_timestep);
            transition_distribution->insert_or_assign(
                static_pointer_cast<const State>(terminal_state), 1.0);
        } else if (cell == 'E') {
            shared_ptr<const ResourceGatheringTerminalState> terminal_state = get_terminal_state(next_timestep);
            transition_distribution->insert_or_assign(
                static_pointer_cast<const State>(terminal_state), ENEMY_DEATH_PROB);
            transition_distribution->insert_or_assign(
                static_pointer_cast<const State>(next_state), 1.0 - ENEMY_DEATH_PROB);
        } else {
            transition_distribution->insert_or_assign(
                static_pointer_cast<const State>(next_state), 1.0);
        }
        
        return transition_distribution;
    }

    shared_ptr<const State> PortedResourceGatheringThtsEnv::sample_transition_distribution(
        shared_ptr<const ResourceGatheringState> state, 
        shared_ptr<const IntAction> action, 
        RandManager& rand_manager,
        ThtsContext& ctx) const 
    {
        shared_ptr<const ResourceGatheringState> next_state = make_next_state(state, action);
        int next_x = get_x(next_state);
        int next_y = get_y(next_state);
        int next_timestep = get_timestep(next_state);
        char cell = get_map_value(next_x, next_y);
        
        if (cell == 'H') {
            return static_pointer_cast<const State>(get_terminal_state(next_timestep));
        } else if (cell == 'E') {
            double rand_val = rand_manager.get_rand_uniform();
            if (rand_val < ENEMY_DEATH_PROB) {
                return static_pointer_cast<const State>(get_pre_death_state(next_timestep));
            } else {
                return static_pointer_cast<const State>(next_state);
            }
        } else {
            return static_pointer_cast<const State>(next_state);
        }
    }

    Eigen::ArrayXd PortedResourceGatheringThtsEnv::get_mo_reward(
        shared_ptr<const ResourceGatheringState> state, 
        shared_ptr<const IntAction> action,
        ThtsContext& ctx) const 
    {
        int dim = this->timed ? 4 : 3;
        Eigen::ArrayXd reward = Eigen::ArrayXd::Zero(dim);

        if (this->timed)
        {
            reward[3] = -1.0;
        }
        
        shared_ptr<const ResourceGatheringState> next_state = make_next_state(state, action);
        int next_x = get_x(next_state);
        int next_y = get_y(next_state);
        char cell = get_map_value(next_x, next_y);
        
        if (cell == 'E') {
            // Enemy: reward depends on outcome
            // If survived: [0, 0, 0]
            // If killed: transition to pre-death state (reward handled there)
        } else if (cell == 'H') {
            int has_gold = get_has_gold(next_state);
            int has_gem = get_has_gem(next_state);
            reward[1] = has_gold;
            reward[2] = has_gem;
        }
        
        return reward;
    }

    // Interface implementations
    shared_ptr<const State> PortedResourceGatheringThtsEnv::get_initial_state_itfc() const {
        shared_ptr<const ResourceGatheringState> init_state = get_initial_state();
        return static_pointer_cast<const State>(init_state);
    }

    bool PortedResourceGatheringThtsEnv::is_sink_state_itfc(shared_ptr<const State> state, ThtsContext& ctx) const {
        return is_sink_state(state, ctx);
    }

    shared_ptr<ActionVector> PortedResourceGatheringThtsEnv::get_valid_actions_itfc(
        shared_ptr<const State> state, ThtsContext& ctx) const
    {
        if (is_sink_state(state, ctx)) {
            return make_shared<ActionVector>();
        }
        
        const ResourceGatheringPreDeathState* pre_death_state = 
            dynamic_cast<const ResourceGatheringPreDeathState*>(state.get());
        if (pre_death_state != nullptr) {
            auto actions = make_shared<ActionVector>();
            actions->push_back(make_shared<const IntAction>(0));
            return actions;
        }
        
        shared_ptr<const ResourceGatheringState> state_itfc = 
            static_pointer_cast<const ResourceGatheringState>(state);
        shared_ptr<IntActionVector> valid_actions_itfc = get_valid_actions(state_itfc, ctx);

        shared_ptr<ActionVector> valid_actions = make_shared<ActionVector>();
        for (shared_ptr<const IntAction> act : *valid_actions_itfc) {
            valid_actions->push_back(static_pointer_cast<const Action>(act));
        }
        return valid_actions;
    }

    shared_ptr<StateDistr> PortedResourceGatheringThtsEnv::get_transition_distribution_itfc(
        shared_ptr<const State> state, shared_ptr<const Action> action, ThtsContext& ctx) const
    {
        if (is_sink_state(state, ctx)) {
            return make_shared<StateDistr>();
        }
        
        const ResourceGatheringPreDeathState* pre_death_state = 
            dynamic_cast<const ResourceGatheringPreDeathState*>(state.get());
        if (pre_death_state != nullptr) {
            shared_ptr<StateDistr> transition_distribution = make_shared<StateDistr>();
            shared_ptr<const ResourceGatheringTerminalState> terminal_state = 
                get_terminal_state(pre_death_state->timestep + 1);
            transition_distribution->insert_or_assign(
                static_pointer_cast<const State>(terminal_state), 1.0);
            return transition_distribution;
        }
        
        shared_ptr<const ResourceGatheringState> state_itfc = 
            static_pointer_cast<const ResourceGatheringState>(state);
        shared_ptr<const IntAction> action_itfc = 
            static_pointer_cast<const IntAction>(action);
        return get_transition_distribution(state_itfc, action_itfc, ctx);
    }

    shared_ptr<const State> PortedResourceGatheringThtsEnv::sample_transition_distribution_itfc(
        shared_ptr<const State> state, 
        shared_ptr<const Action> action, 
        RandManager& rand_manager, 
        ThtsContext& ctx) const 
    {
        if (is_sink_state(state, ctx)) {
            return state;
        }
        
        const ResourceGatheringPreDeathState* pre_death_state = 
            dynamic_cast<const ResourceGatheringPreDeathState*>(state.get());
        if (pre_death_state != nullptr) {
            return static_pointer_cast<const State>(
                get_terminal_state(pre_death_state->timestep + 1));
        }
        
        shared_ptr<const ResourceGatheringState> state_itfc = 
            static_pointer_cast<const ResourceGatheringState>(state);
        shared_ptr<const IntAction> action_itfc = 
            static_pointer_cast<const IntAction>(action);
        return sample_transition_distribution(state_itfc, action_itfc, rand_manager, ctx);
    }

    std::shared_ptr<ObservationDistr> PortedResourceGatheringThtsEnv::get_observation_distribution_itfc(
        std::shared_ptr<const Action> action, 
        std::shared_ptr<const State> next_state, 
        ThtsContext& ctx) const
    {
        return thts::ThtsEnv::get_observation_distribution_itfc(action, next_state, ctx);
    }

    std::shared_ptr<const Observation> PortedResourceGatheringThtsEnv::sample_observation_distribution_itfc(
        std::shared_ptr<const Action> action, 
        std::shared_ptr<const State> next_state, 
        RandManager& rand_manager,
        ThtsContext& ctx) const 
    {
        return thts::ThtsEnv::sample_observation_distribution_itfc(action, next_state, rand_manager, ctx);
    }

    Eigen::ArrayXd PortedResourceGatheringThtsEnv::get_mo_reward_itfc(
        shared_ptr<const State> state, 
        shared_ptr<const Action> action,
        ThtsContext& ctx) const
    {   
        const ResourceGatheringPreDeathState* pre_death = 
            dynamic_cast<const ResourceGatheringPreDeathState*>(state.get());
        if (pre_death != nullptr) {
            int dim = this->timed ? 4 : 3;
            Eigen::ArrayXd reward = Eigen::ArrayXd::Zero(dim);
            reward[0] = -1.0;
            return reward;
        }

        shared_ptr<const ResourceGatheringState> state_itfc = 
            static_pointer_cast<const ResourceGatheringState>(state);
        shared_ptr<const IntAction> action_itfc = 
            static_pointer_cast<const IntAction>(action);
        return get_mo_reward(state_itfc, action_itfc, ctx);
    }
}

// Hash and equality implementations
namespace std {
    using namespace thts;

    size_t hash<shared_ptr<const ResourceGatheringState>>::operator()(
        const shared_ptr<const ResourceGatheringState>& state) const 
    {
        return state->hash();
    }
    
    bool operator==(const shared_ptr<const ResourceGatheringState>& lhs, 
                    const shared_ptr<const ResourceGatheringState>& rhs) {
        return lhs->equals_itfc(*rhs);
    }

    bool equal_to<shared_ptr<const ResourceGatheringState>>::operator()(
        const shared_ptr<const ResourceGatheringState>& lhs, 
        const shared_ptr<const ResourceGatheringState>& rhs) const 
    {
        return lhs->equals_itfc(*rhs);
    }

    size_t hash<shared_ptr<const ResourceGatheringPreDeathState>>::operator()(
        const shared_ptr<const ResourceGatheringPreDeathState>& state) const 
    {
        return state->hash();
    }
    
    bool operator==(const shared_ptr<const ResourceGatheringPreDeathState>& lhs, 
                    const shared_ptr<const ResourceGatheringPreDeathState>& rhs) {
        return lhs->equals_itfc(*rhs);
    }

    bool equal_to<shared_ptr<const ResourceGatheringPreDeathState>>::operator()(
        const shared_ptr<const ResourceGatheringPreDeathState>& lhs, 
        const shared_ptr<const ResourceGatheringPreDeathState>& rhs) const 
    {
        return lhs->equals_itfc(*rhs);
    }

    size_t hash<shared_ptr<const ResourceGatheringTerminalState>>::operator()(
        const shared_ptr<const ResourceGatheringTerminalState>& state) const 
    {
        return state->hash();
    }
    
    bool operator==(const shared_ptr<const ResourceGatheringTerminalState>& lhs, 
                    const shared_ptr<const ResourceGatheringTerminalState>& rhs) {
        return lhs->equals_itfc(*rhs);
    }

    bool equal_to<shared_ptr<const ResourceGatheringTerminalState>>::operator()(
        const shared_ptr<const ResourceGatheringTerminalState>& lhs, 
        const shared_ptr<const ResourceGatheringTerminalState>& rhs) const 
    {
        return lhs->equals_itfc(*rhs);
    }
}
