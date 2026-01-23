#pragma once

#include "thts_types.h"
#include "mo/mo_thts_env.h"
#include "custom_deep_sea_treasure_maps.h"
#include <vector>
#include <string>

namespace thts {
    using namespace std;

    /**
     * Custom state class for Deep Sea Treasure environment.
     * Contains named parameters x, y, and timestep.
     */
    class DSTState : public State {
        public:
            int x;        // Row coordinate
            int y;        // Column coordinate
            int timestep; // Current timestep
            
            DSTState(int x, int y, int timestep) : x(x), y(y), timestep(timestep) {}
            virtual ~DSTState() = default;
            
            virtual std::size_t hash() const override;
            bool equals(const DSTState& other) const;
            virtual bool equals_itfc(const Observation& other) const override;
            virtual std::string get_pretty_print_string() const override;
    };

    /**
     * A C++ port of the DeepSeaTreasure environment from mo-gymnasium.
     * 
     * State: DSTState with x, y, timestep where:
     *   - x: row coordinate (0-10)
     *   - y: column coordinate (0-10 for convex/concave, 0-19 for mirrored)
     *   - timestep: current timestep (0 to max_timestep)
     * 
     * Actions: 0=up, 1=down, 2=left, 3=right
     * 
     * Rewards: 2-dimensional [treasure_value, time_penalty=-1]
     * 
     * Termination: when reaching a treasure (non-zero, non--10 value in map)
     *              or when timestep reaches max_timestep (sink state)
     */
    class PortedDeepSeaTreasureThtsEnv : public MoThtsEnv {
        private:
            static constexpr int TERMINAL_STATE_POS = -100;
            
            int map_id;
            double swept_by_current_prob;
            int max_timestep;
            vector<vector<double>> sea_map;
            int num_cols;
            int num_rows;
            
            // Direction vectors: [dx, dy] for actions 0-3
            static constexpr int DIR[4][2] = {
                {-1, 0},  // 0: up
                {1, 0},   // 1: down
                {0, -1},  // 2: left
                {0, 1}    // 3: right
            };
            
            // Initial position (always top-left for standard maps)
            pair<int, int> get_initial_position() const {
                return make_pair(0, 0);
            }

        public:
            /**
             * Constructor with map selection by ID
             * @param map_id: Map ID from custom_deep_sea_treasure_maps (0=default, 1=mogymnasium convex, etc.)
             * @param swept_by_current_prob: Probability of being swept by current (additional random action)
             * @param max_timestep: Maximum timestep; states at this timestep are sink states
             */
            PortedDeepSeaTreasureThtsEnv(int map_id = 1, double swept_by_current_prob = 0.0, int max_timestep = 1000);
            
            PortedDeepSeaTreasureThtsEnv(PortedDeepSeaTreasureThtsEnv& other);

            virtual ~PortedDeepSeaTreasureThtsEnv() = default;

            virtual std::shared_ptr<ThtsEnv> clone() override;

            // Helper methods
            int get_x(shared_ptr<const DSTState> state) const {
                return state->x;
            }

            int get_y(shared_ptr<const DSTState> state) const {
                return state->y;
            }

            int get_timestep(shared_ptr<const DSTState> state) const {
                return state->timestep;
            }

            double get_map_value(int x, int y) const;

            bool is_valid_state(int x, int y) const;

            bool is_treasure_cell(int x, int y) const;

            shared_ptr<const DSTState> get_initial_state() const;
            shared_ptr<const DSTState> get_terminal_state() const;

            bool is_sink_state(shared_ptr<const DSTState> state, ThtsContext& ctx) const;
            bool is_terminal_state(shared_ptr<const DSTState> state) const;

            shared_ptr<IntActionVector> get_valid_actions(
                shared_ptr<const DSTState> state, ThtsContext& ctx) const;
            
            /**
             * Returns the set of all possible reachable states in the environment
             * Includes all non-rock states and the terminal state
             */
            std::unordered_set<std::shared_ptr<const State>> get_all_states() const;

        private:
            shared_ptr<const DSTState> sample_next_state(
                shared_ptr<const DSTState> state, 
                shared_ptr<const IntAction> action,
                RandManager& rand_manager) const;

            void initialize_map(int map_id);

        public:
            shared_ptr<StateDistr> get_transition_distribution(
                shared_ptr<const DSTState> state, 
                shared_ptr<const IntAction> action, 
                ThtsContext& ctx) const;

            shared_ptr<const State> sample_transition_distribution(
                shared_ptr<const DSTState> state, 
                shared_ptr<const IntAction> action, 
                RandManager& rand_manager,
                ThtsContext& ctx) const;

            Eigen::ArrayXd get_mo_reward(
                shared_ptr<const DSTState> state, 
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

