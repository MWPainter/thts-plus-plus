#pragma once

#include "thts_types.h"
#include "mo/mo_thts_env.h"
#include <array>

namespace thts {
    using namespace std;

    /**
     * State representation for ResourceGathering: [x, y, has_gold, has_gem]
     */
    class ResourceGatheringState : public State {
        public:
            std::array<int, 4> state;  // [x, y, has_gold, has_gem]

            ResourceGatheringState(int x, int y, int has_gold, int has_gem) 
                : state({x, y, has_gold, has_gem}) {}
            
            ResourceGatheringState(std::array<int, 4> state) : state(state) {}
            
            virtual ~ResourceGatheringState() = default;
            virtual std::size_t hash() const override;
            bool equals(const ResourceGatheringState& other) const;
            virtual bool equals_itfc(const Observation& other) const override;
            virtual std::string get_pretty_print_string() const override;
    };

    /**
     * Pre-death state for ResourceGathering (intermediate state before terminal)
     */
    class ResourceGatheringPreDeathState : public State {
        public:
            ResourceGatheringPreDeathState() {}
            virtual ~ResourceGatheringPreDeathState() = default;
            virtual std::size_t hash() const override;
            bool equals(const ResourceGatheringPreDeathState& other) const;
            virtual bool equals_itfc(const Observation& other) const override;
            virtual std::string get_pretty_print_string() const override;
    };

    /**
     * Terminal state for ResourceGathering
     */
    class ResourceGatheringTerminalState : public State {
        public:
            ResourceGatheringTerminalState() {}
            virtual ~ResourceGatheringTerminalState() = default;
            virtual std::size_t hash() const override;
            bool equals(const ResourceGatheringTerminalState& other) const;
            virtual bool equals_itfc(const Observation& other) const override;
            virtual std::string get_pretty_print_string() const override;
    };

    typedef std::unordered_map<std::shared_ptr<const ResourceGatheringState>, double> ResourceGatheringStateDistr;

    /**
     * A C++ port of the ResourceGathering environment from mo-gymnasium.
     * 
     * State: [x, y, has_gold, has_gem] where:
     *   - x, y: position coordinates (0-4)
     *   - has_gold: 0 or 1
     *   - has_gem: 0 or 1
     * 
     * Actions: 0=up, 1=down, 2=left, 3=right
     * 
     * Rewards: 3-dimensional [enemy_killed, gold_collected, gem_collected]
     */
    class PortedResourceGatheringThtsEnv : public MoThtsEnv {
        private:
            static constexpr int SIZE = 5;
            static constexpr double ENEMY_DEATH_PROB = 0.1;
            
            // Map layout: [row][col] - stored as single characters
            static constexpr char MAP[SIZE][SIZE] = {
                {' ', ' ', 'R', 'E', ' '},
                {' ', ' ', 'E', ' ', 'R'},
                {' ', ' ', ' ', ' ', ' '},
                {' ', ' ', ' ', ' ', ' '},
                {' ', ' ', 'H', ' ', ' '}
            };
            
            // Helper to get full cell identifier (R1, R2, E1, E2)
            char get_cell_type(int x, int y) const;
            
            static constexpr int INITIAL_X = 4;
            static constexpr int INITIAL_Y = 2;
            
            // Direction vectors: [dx, dy] for actions 0-3
            static constexpr int DIR[4][2] = {
                {-1, 0},  // 0: up
                {1, 0},   // 1: down
                {0, -1},  // 2: left
                {0, 1}    // 3: right
            };

        public:
            PortedResourceGatheringThtsEnv() : 
                ThtsEnv(true),
                MoThtsEnv(3, true)  // reward_dim=3, fully_observable=true
            {
            }

            PortedResourceGatheringThtsEnv(PortedResourceGatheringThtsEnv& other) : 
                ThtsEnv(true),
                MoThtsEnv(3, true)
            {
            }

            virtual ~PortedResourceGatheringThtsEnv() = default;

            virtual std::shared_ptr<ThtsEnv> clone() override;

            // Helper methods
            int get_x(shared_ptr<const ResourceGatheringState> state) const;
            int get_y(shared_ptr<const ResourceGatheringState> state) const;
            int get_has_gold(shared_ptr<const ResourceGatheringState> state) const;
            int get_has_gem(shared_ptr<const ResourceGatheringState> state) const;
            bool is_valid_position(int x, int y) const;
            char get_map_value(int x, int y) const;
            shared_ptr<const ResourceGatheringState> get_initial_state() const;
            shared_ptr<const ResourceGatheringPreDeathState> get_pre_death_state() const;
            shared_ptr<const ResourceGatheringTerminalState> get_terminal_state() const;
            bool is_sink_state(shared_ptr<const State> state, ThtsContext& ctx) const;
            shared_ptr<IntActionVector> get_valid_actions(
                shared_ptr<const ResourceGatheringState> state, ThtsContext& ctx) const;
            
            /**
             * Returns the set of all possible states in the environment
             * Includes all regular states, pre-death state, and terminal state
             */
            std::unordered_set<std::shared_ptr<const State>> get_all_states() const;

        private:
            shared_ptr<const ResourceGatheringState> make_next_state(
                shared_ptr<const ResourceGatheringState> state, 
                shared_ptr<const IntAction> action) const;

        public:
            shared_ptr<StateDistr> get_transition_distribution(
                shared_ptr<const ResourceGatheringState> state, 
                shared_ptr<const IntAction> action, 
                ThtsContext& ctx) const;

            shared_ptr<const State> sample_transition_distribution(
                shared_ptr<const ResourceGatheringState> state, 
                shared_ptr<const IntAction> action, 
                RandManager& rand_manager,
                ThtsContext& ctx) const;

            Eigen::ArrayXd get_mo_reward(
                shared_ptr<const ResourceGatheringState> state, 
                shared_ptr<const IntAction> action,
                ThtsContext& ctx) const;

            // Interface implementations
            virtual shared_ptr<const State> get_initial_state_itfc() const override;
            virtual bool is_sink_state_itfc(shared_ptr<const State> state, ThtsContext& ctx) const override;
            virtual shared_ptr<ActionVector> get_valid_actions_itfc(
                shared_ptr<const State> state, ThtsContext& ctx) const override;
            virtual shared_ptr<StateDistr> get_transition_distribution_itfc(
                shared_ptr<const State> state, shared_ptr<const Action> action, ThtsContext& ctx) const override;
            virtual shared_ptr<const State> sample_transition_distribution_itfc(
                shared_ptr<const State> state, 
                shared_ptr<const Action> action, 
                RandManager& rand_manager, 
                ThtsContext& ctx) const override;
            virtual std::shared_ptr<ObservationDistr> get_observation_distribution_itfc(
                std::shared_ptr<const Action> action, 
                std::shared_ptr<const State> next_state, 
                ThtsContext& ctx) const override;
            virtual std::shared_ptr<const Observation> sample_observation_distribution_itfc(
                std::shared_ptr<const Action> action, 
                std::shared_ptr<const State> next_state, 
                RandManager& rand_manager,
                ThtsContext& ctx) const override;
            virtual Eigen::ArrayXd get_mo_reward_itfc(
                shared_ptr<const State> state, 
                shared_ptr<const Action> action,
                ThtsContext& ctx) const override;
    };
}

// Hash and equality for ResourceGatheringState and ResourceGatheringTerminalState
namespace std {
    using namespace thts;

    template <> 
    struct hash<shared_ptr<const ResourceGatheringState>> {
        size_t operator()(const shared_ptr<const ResourceGatheringState>& state) const;
    };
    
    bool operator==(const shared_ptr<const ResourceGatheringState>& lhs, 
                    const shared_ptr<const ResourceGatheringState>& rhs);

    template <> 
    struct equal_to<shared_ptr<const ResourceGatheringState>> {
        bool operator()(const shared_ptr<const ResourceGatheringState>& lhs, 
                       const shared_ptr<const ResourceGatheringState>& rhs) const;
    };

    template <> 
    struct hash<shared_ptr<const ResourceGatheringPreDeathState>> {
        size_t operator()(const shared_ptr<const ResourceGatheringPreDeathState>& state) const;
    };
    
    bool operator==(const shared_ptr<const ResourceGatheringPreDeathState>& lhs, 
                    const shared_ptr<const ResourceGatheringPreDeathState>& rhs);

    template <> 
    struct equal_to<shared_ptr<const ResourceGatheringPreDeathState>> {
        bool operator()(const shared_ptr<const ResourceGatheringPreDeathState>& lhs, 
                       const shared_ptr<const ResourceGatheringPreDeathState>& rhs) const;
    };

    template <> 
    struct hash<shared_ptr<const ResourceGatheringTerminalState>> {
        size_t operator()(const shared_ptr<const ResourceGatheringTerminalState>& state) const;
    };
    
    bool operator==(const shared_ptr<const ResourceGatheringTerminalState>& lhs, 
                    const shared_ptr<const ResourceGatheringTerminalState>& rhs);

    template <> 
    struct equal_to<shared_ptr<const ResourceGatheringTerminalState>> {
        bool operator()(const shared_ptr<const ResourceGatheringTerminalState>& lhs, 
                       const shared_ptr<const ResourceGatheringTerminalState>& rhs) const;
    };
}

