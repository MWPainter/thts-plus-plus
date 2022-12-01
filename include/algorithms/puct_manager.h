#pragma once

#include "uct_manager.h"

namespace thts {
    /**
     * Args object so that params can be set in a more named args way
     */
    struct PuctManagerArgs : public UctManagerArgs {
        static constexpr double puct_power_default=0.5;

        double puct_power;

        PuctManagerArgs(std::shared_ptr<ThtsEnv> thts_env) :
            UctManagerArgs(thts_env),
            puct_power(puct_power_default) {}

        virtual ~PuctManagerArgs() = default;
    };

    /**
     * A specific instance of ThtsManager for PUCT algorithms.
     * 
     * Member variables:
     *      puct_power:
     *          The power to use in 
     */
    class PuctManager : public UctManager {
        public:
            double puct_power;

            PuctManager(PuctManagerArgs& args) :
                UctManager(args),
                puct_power(args.puct_power) {};

            PuctManager(
                std::shared_ptr<ThtsEnv> thts_env,
                int max_depth=PuctManagerArgs::max_depth_default,
                double bias=PuctManagerArgs::bias_default,
                double puct_power=PuctManagerArgs::puct_power_default,
                int heuristic_psuedo_trials=PuctManagerArgs::heuristic_psuedo_trials_default,
                HeuristicFnPtr heuristic_fn=helper::zero_heuristic_fn,
                PriorFnPtr prior_fn=nullptr,
                bool mcts_mode=PuctManagerArgs::mcts_mode_default, 
                bool use_transposition_table=PuctManagerArgs::use_transposition_table_default, 
                int num_transposition_table_mutexes=PuctManagerArgs::num_transposition_table_mutexes_default,
                bool is_two_player_game=PuctManagerArgs::is_two_player_game_default,
                double epsilon_exploration=PuctManagerArgs::epsilon_exploration_default,
                bool recommend_most_visited=PuctManagerArgs::recommend_most_visited_default,
                int seed=PuctManagerArgs::seed_default) :
                    UctManager(
                        thts_env,
                        max_depth,
                        bias,
                        heuristic_psuedo_trials,
                        heuristic_fn,
                        prior_fn,
                        mcts_mode,
                        use_transposition_table,
                        num_transposition_table_mutexes,
                        is_two_player_game,
                        epsilon_exploration,
                        recommend_most_visited,
                        seed),
                    puct_power(puct_power) {};
    };
}