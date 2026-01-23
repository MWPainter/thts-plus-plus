#include "ported_dst.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <sstream>

using namespace std;

namespace thts {
    /**
     * DSTState implementation
     */
    std::size_t DSTState::hash() const {
        // Combine hashes of x, y, and timestep
        std::size_t h1 = std::hash<int>{}(x);
        std::size_t h2 = std::hash<int>{}(y);
        std::size_t h3 = std::hash<int>{}(timestep);
        // Combine using a common technique
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }

    bool DSTState::equals(const DSTState& other) const {
        return x == other.x && y == other.y && timestep == other.timestep;
    }

    bool DSTState::equals_itfc(const Observation& other) const {
        const DSTState* other_state = dynamic_cast<const DSTState*>(&other);
        if (other_state == nullptr) return false;
        return equals(*other_state);
    }

    std::string DSTState::get_pretty_print_string() const {
        std::ostringstream oss;
        oss << "DSTState(x=" << x << ", y=" << y << ", timestep=" << timestep << ")";
        return oss.str();
    }

    /**
     * Constructor
     */
    PortedDeepSeaTreasureThtsEnv::PortedDeepSeaTreasureThtsEnv(int map_id, double swept_by_current_prob, int max_timestep) : 
        ThtsEnv(true),
        MoThtsEnv(2, true),  // reward_dim=2, fully_observable=true
        map_id(map_id),
        swept_by_current_prob(swept_by_current_prob),
        max_timestep(max_timestep)
    {
        initialize_map(map_id);
    }

    PortedDeepSeaTreasureThtsEnv::PortedDeepSeaTreasureThtsEnv(PortedDeepSeaTreasureThtsEnv& other) : 
        ThtsEnv(true),
        MoThtsEnv(2, true),
        map_id(other.map_id),
        swept_by_current_prob(other.swept_by_current_prob),
        max_timestep(other.max_timestep),
        sea_map(other.sea_map),
        num_cols(other.num_cols),
        num_rows(other.num_rows)
    {
    }

    void PortedDeepSeaTreasureThtsEnv::initialize_map(int map_id) {
        const TreasureMap* treasure_map = get_map(map_id);
        if (treasure_map == nullptr) {
            throw invalid_argument("Invalid map_id: " + to_string(map_id));
        }
        
        // Treasure maps are sorted: for each i, there is exactly one entry with x=i at position i
        // Find maximum coordinates to determine grid size
        int max_x = treasure_map->size() - 1;  // Last index is the max x
        int max_y = 0;
        for (const auto& treasure : *treasure_map) {
            max_y = max(max_y, treasure.position[1]);
        }
        
        // Determine grid dimensions
        num_cols = max_x + 1;
        num_rows = max_y + 1;
        
        // Initialize sea_map with zeros
        sea_map = vector<vector<double>>(num_cols, vector<double>(num_rows, 0.0));
        
        // Place treasures using the sorted property
        // For each row x, treasure_map[x] contains the treasure for that row
        for (int x = 0; x < num_cols; x++) {
            const TreasureEntry& treasure = (*treasure_map)[x];
            int treasure_y = treasure.position[1];
            
            sea_map[x][treasure_y] = treasure.value;
            for (int y = treasure_y + 1; y < num_rows; y++) {
                sea_map[x][y] = -10.0;
            }
        }
    }

    std::shared_ptr<ThtsEnv> PortedDeepSeaTreasureThtsEnv::clone() {
        return std::dynamic_pointer_cast<ThtsEnv>(
            std::make_shared<PortedDeepSeaTreasureThtsEnv>(*this));
    }


    double PortedDeepSeaTreasureThtsEnv::get_map_value(int x, int y) const {
        if (x < 0 || x >= num_cols || y < 0 || y >= num_rows) {
            return -10.0;  // Out of bounds treated as rock
        }
        return sea_map[x][y];
    }

    bool PortedDeepSeaTreasureThtsEnv::is_valid_state(int x, int y) const {
        // Check bounds
        if (x < 0 || x >= num_cols || y < 0 || y >= num_rows) {
            return false;
        }
        // Check if it's a rock
        return get_map_value(x, y) != -10.0;
    }

    bool PortedDeepSeaTreasureThtsEnv::is_treasure_cell(int x, int y) const {
        double value = get_map_value(x, y);
        return value != 0.0 && value != -10.0;
    }

    shared_ptr<const DSTState> PortedDeepSeaTreasureThtsEnv::get_initial_state() const {
        pair<int, int> init_pos = get_initial_position();
        return make_shared<DSTState>(init_pos.first, init_pos.second, 0);
    }

    shared_ptr<const DSTState> PortedDeepSeaTreasureThtsEnv::get_terminal_state() const {
        return make_shared<DSTState>(TERMINAL_STATE_POS, TERMINAL_STATE_POS, -1);
    }

    bool PortedDeepSeaTreasureThtsEnv::is_terminal_state(shared_ptr<const DSTState> state) const {
        int x = get_x(state);
        int y = get_y(state);
        return (x == TERMINAL_STATE_POS && y == TERMINAL_STATE_POS);
    }

    bool PortedDeepSeaTreasureThtsEnv::is_sink_state(shared_ptr<const DSTState> state, ThtsContext& ctx) const {
        // State is a sink if it's the terminal state 
        // OR if timestep has reached max_timestep
        // AND if not at a treasure location (because we need to perform dummy transition to terminal state)
        return is_terminal_state(state) || ((get_timestep(state) >= max_timestep) && !is_treasure_cell(get_x(state), get_y(state)));
    }

    shared_ptr<IntActionVector> PortedDeepSeaTreasureThtsEnv::get_valid_actions(
        shared_ptr<const DSTState> state, ThtsContext& ctx) const 
    {
        shared_ptr<IntActionVector> valid_actions = make_shared<IntActionVector>();
        // Sink states (terminal or max timestep) have no valid actions
        if (is_sink_state(state, ctx)) {
            return valid_actions;
        }
        // If at a treasure location, all actions just transition to terminal state, so just reutrn one
        if (is_treasure_cell(get_x(state), get_y(state))) {
            valid_actions->push_back(make_shared<const IntAction>(0));
            return valid_actions;
        }
        // For normal states, all 4 actions are always valid (invalid moves just don't change position)
        for (int i = 0; i < 4; i++) {
            valid_actions->push_back(make_shared<const IntAction>(i));
        }
        return valid_actions;
    }

    unordered_set<shared_ptr<const State>> PortedDeepSeaTreasureThtsEnv::get_all_states() const {
        unordered_set<shared_ptr<const State>> all_states;
        
        // Add all valid (non-rock) positions in the grid for all timesteps
        for (int t = 0; t <= max_timestep; t++) {
            for (int x = 0; x < num_cols; x++) {
                for (int y = 0; y < num_rows; y++) {
                    if (is_valid_state(x, y)) {
                        all_states.insert(
                            static_pointer_cast<const State>(
                                make_shared<DSTState>(x, y, t)));
                    }
                }
            }
        }
        
        // Add terminal state
        all_states.insert(
            static_pointer_cast<const State>(get_terminal_state()));
        
        return all_states;
    }

    shared_ptr<const DSTState> PortedDeepSeaTreasureThtsEnv::sample_next_state(
        shared_ptr<const DSTState> state, 
        shared_ptr<const IntAction> action,
        RandManager& rand_manager) const
    {
        int x = get_x(state);
        int y = get_y(state);
        int t = get_timestep(state);
        
        // Compute next position from the action
        int next_x = x + DIR[action->action][0];
        int next_y = y + DIR[action->action][1];
        
        // Apply current sweep if probability > 0
        if (swept_by_current_prob > 0.0) {
            double rand_val = rand_manager.get_rand_uniform();
            if (rand_val < swept_by_current_prob) {
                // Sample random direction to apply
                int random_direction = rand_manager.get_rand_int(0, 3);  // Random action 0-3
                next_x += DIR[random_direction][0];
                next_y += DIR[random_direction][1];
            }
        }
        
        // Check if valid move
        if (!is_valid_state(next_x, next_y)) {
            next_x = x;
            next_y = y;
        }
        
        // Increment timestep
        return make_shared<DSTState>(next_x, next_y, t + 1);
    }

    shared_ptr<StateDistr> PortedDeepSeaTreasureThtsEnv::get_transition_distribution(
        shared_ptr<const DSTState> state, 
        shared_ptr<const IntAction> action, 
        ThtsContext& ctx) const 
    {
        shared_ptr<StateDistr> transition_distribution = make_shared<StateDistr>();
        
        int x = get_x(state);
        int y = get_y(state);
        int t = get_timestep(state);
        
        // If already in a sink state (terminal or max timestep), return empty distribution
        if (is_sink_state(state, ctx)) {
            return transition_distribution;
        }
        
        // If at a treasure location, any action transitions to terminal state
        if (is_treasure_cell(x, y)) {
            shared_ptr<const DSTState> terminal_state = get_terminal_state();
            transition_distribution->insert_or_assign(
                static_pointer_cast<const State>(terminal_state), 1.0);
            return transition_distribution;
        }

        // Increment timestep for the next state
        int next_t = t + 1;

        // First, compute position after action
        int next_x = x + DIR[action->action][0];
        int next_y = y + DIR[action->action][1];
        if (!is_valid_state(next_x, next_y)) {
            next_x = x;
            next_y = y;
        }
        
        // With probability (1 - swept_by_current_prob): stay at next_x, next_y
        transition_distribution->insert_or_assign(
            static_pointer_cast<const State>(make_shared<DSTState>(next_x, next_y, next_t)),
            1.0 - swept_by_current_prob);
        
        // Otherwise, normal movement (with potential current sweep)
        // For deterministic distribution, we need to consider all possible outcomes
        // If swept_by_current_prob > 0, we have stochastic transitions
        if (swept_by_current_prob > 0.0) {
            
            // With probability swept_by_current_prob: apply random action
            // Each of the 4 random actions has equal probability
            double prob_per_action = swept_by_current_prob / 4.0;
            for (int random_action = 0; random_action < 4; random_action++) 
            {
                int swept_x = next_x + DIR[random_action][0];
                int swept_y = next_y + DIR[random_action][1];

                if (!is_valid_state(swept_x, swept_y)) {
                    swept_x = x;
                    swept_y = y;
                }
                
                shared_ptr<const DSTState> swept_state = 
                    make_shared<DSTState>(swept_x, swept_y, next_t);
                auto it = transition_distribution->find(static_pointer_cast<const State>(swept_state));
                if (it != transition_distribution->end()) 
                {
                    // If this state already exists (e.g., random action = no-op), add probability
                    // This can happen at the walls of the map for example
                    transition_distribution->insert_or_assign(it->first, it->second + prob_per_action);
                } 
                else 
                {
                    transition_distribution->insert_or_assign(
                        static_pointer_cast<const State>(swept_state), prob_per_action);
                }
            }
        } 
        
        return transition_distribution;
    }

    shared_ptr<const State> PortedDeepSeaTreasureThtsEnv::sample_transition_distribution(
        shared_ptr<const DSTState> state, 
        shared_ptr<const IntAction> action, 
        RandManager& rand_manager,
        ThtsContext& ctx) const 
    {
        int x = get_x(state);
        int y = get_y(state);
        
        // If already in a sink state (terminal or max timestep), return state as-is
        if (is_sink_state(state, ctx)) {
            return static_pointer_cast<const State>(state);
        }
        
        // If at a treasure location, any action transitions to terminal state
        if (is_treasure_cell(x, y)) {
            return static_pointer_cast<const State>(get_terminal_state());
        }
        
        // Otherwise, normal movement (with potential current sweep)
        shared_ptr<const DSTState> next_state = sample_next_state(state, action, rand_manager);
        return static_pointer_cast<const State>(next_state);
    }

    Eigen::ArrayXd PortedDeepSeaTreasureThtsEnv::get_mo_reward(
        shared_ptr<const DSTState> state, 
        shared_ptr<const IntAction> action,
        ThtsContext& ctx) const 
    {
        Eigen::ArrayXd reward = Eigen::ArrayXd::Zero(2);
        
        int x = get_x(state);
        int y = get_y(state);
        
        // If transitioning from treasure to terminal, give treasure reward
        if (is_treasure_cell(x, y)) {
            double treasure_value = get_map_value(x, y);
            reward[0] = treasure_value;
            reward[1] = 0.0;  // No Time penalty - fake transition
            return reward;
        }
        
        // Otherwise, normal movement (no treasure reward)
        reward[0] = 0.0;
        reward[1] = -1.0;  // Time penalty
        
        return reward;
    }

    // Interface implementations
    shared_ptr<const State> PortedDeepSeaTreasureThtsEnv::get_initial_state_itfc() const {
        shared_ptr<const DSTState> init_state = get_initial_state();
        return static_pointer_cast<const State>(init_state);
    }

    bool PortedDeepSeaTreasureThtsEnv::is_sink_state_itfc(shared_ptr<const State> state, ThtsContext& ctx) const {
        shared_ptr<const DSTState> state_itfc = static_pointer_cast<const DSTState>(state);
        return is_sink_state(state_itfc, ctx);
    }

    shared_ptr<ActionVector> PortedDeepSeaTreasureThtsEnv::get_valid_actions_itfc(
        shared_ptr<const State> state, ThtsContext& ctx) const
    {
        shared_ptr<const DSTState> state_itfc = static_pointer_cast<const DSTState>(state);
        shared_ptr<IntActionVector> valid_actions_itfc = get_valid_actions(state_itfc, ctx);

        shared_ptr<ActionVector> valid_actions = make_shared<ActionVector>();
        for (shared_ptr<const IntAction> act : *valid_actions_itfc) {
            valid_actions->push_back(static_pointer_cast<const Action>(act));
        }
        return valid_actions;
    }

    shared_ptr<StateDistr> PortedDeepSeaTreasureThtsEnv::get_transition_distribution_itfc(
        shared_ptr<const State> state, shared_ptr<const Action> action, ThtsContext& ctx) const
    {
        shared_ptr<const DSTState> state_itfc = static_pointer_cast<const DSTState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        return get_transition_distribution(state_itfc, action_itfc, ctx);
    }

    shared_ptr<const State> PortedDeepSeaTreasureThtsEnv::sample_transition_distribution_itfc(
        shared_ptr<const State> state, 
        shared_ptr<const Action> action, 
        RandManager& rand_manager, 
        ThtsContext& ctx) const 
    {
        shared_ptr<const DSTState> state_itfc = static_pointer_cast<const DSTState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        return sample_transition_distribution(state_itfc, action_itfc, rand_manager, ctx);
    }

    std::shared_ptr<ObservationDistr> PortedDeepSeaTreasureThtsEnv::get_observation_distribution_itfc(
        std::shared_ptr<const Action> action, 
        std::shared_ptr<const State> next_state, 
        ThtsContext& ctx) const
    {
        return thts::ThtsEnv::get_observation_distribution_itfc(action, next_state, ctx);
    }

    std::shared_ptr<const Observation> PortedDeepSeaTreasureThtsEnv::sample_observation_distribution_itfc(
        std::shared_ptr<const Action> action, 
        std::shared_ptr<const State> next_state, 
        RandManager& rand_manager,
        ThtsContext& ctx) const 
    {
        return thts::ThtsEnv::sample_observation_distribution_itfc(action, next_state, rand_manager, ctx);
    }

    Eigen::ArrayXd PortedDeepSeaTreasureThtsEnv::get_mo_reward_itfc(
        shared_ptr<const State> state, 
        shared_ptr<const Action> action,
        ThtsContext& ctx) const
    {
        shared_ptr<const DSTState> state_itfc = static_pointer_cast<const DSTState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        return get_mo_reward(state_itfc, action_itfc, ctx);
    }
}

