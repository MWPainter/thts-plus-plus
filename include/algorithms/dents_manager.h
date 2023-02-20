#pragma once

#include "ments_manager.h"

namespace thts {
    /**
     * Args object so that params can be set in a more named args way
     */
    struct DentsManagerArgs : public MentsManagerArgs {
        static constexpr double min_temp_default=1.0e-6;

        double min_temp;

        DentsManagerArgs(std::shared_ptr<ThtsEnv> thts_env) :
            MentsManagerArgs(thts_env),
            min_temp(min_temp_default) {}

        virtual ~DentsManagerArgs() = default;
    };

    /**
     * A specific instance of ThtsManager for DENTS algorithms.
     * 
     * Member variables:
     *      min_temp:
     *          The minimum temperature to possibly use
     */
    class DentsManager : public MentsManager {
        public:
            double min_temp;

            DentsManager(DentsManagerArgs& args) :
                MentsManager(args),
                min_temp(args.min_temp) {};

            DentsManager(
                std::shared_ptr<ThtsEnv> thts_env,
                int max_depth=DentsManagerArgs::max_depth_default,
                double temp=DentsManagerArgs::temp_default,
                double min_temp=DentsManagerArgs::min_temp_default,
                double default_q_value=DentsManagerArgs::default_q_value_default,
                HeuristicFnPtr heuristic_fn=nullptr,
                bool use_prior_shift=DentsManagerArgs::use_prior_shift_default,
                double prior_policy_boost=DentsManagerArgs::prior_policy_boost_default,
                double prior_policy_search_weight=DentsManagerArgs::prior_policy_search_weight_default,
                PriorFnPtr prior_fn=nullptr,
                bool mcts_mode=DentsManagerArgs::mcts_mode_default, 
                bool is_two_player_game=DentsManagerArgs::is_two_player_game_default,
                bool use_transposition_table=DentsManagerArgs::use_transposition_table_default, 
                int num_transposition_table_mutexes=DentsManagerArgs::num_transposition_table_mutexes_default,
                double epsilon=DentsManagerArgs::epsilon_default,
                double root_node_extra_epsilon=DentsManagerArgs::root_node_extra_epsilon_default,
                double max_explore_prob=DentsManagerArgs::max_explore_prob_default,
                bool recommend_visit_threshold=DentsManagerArgs::recommend_visit_threshold_default,
                int seed=MentsManagerArgs::seed_default) :
                    MentsManager(
                        thts_env,
                        max_depth,
                        temp,
                        default_q_value,
                        heuristic_fn,
                        use_prior_shift,
                        prior_policy_boost,
                        prior_policy_search_weight,
                        prior_fn,
                        mcts_mode,
                        is_two_player_game,
                        use_transposition_table,
                        num_transposition_table_mutexes,
                        epsilon,
                        root_node_extra_epsilon,
                        max_explore_prob,
                        recommend_visit_threshold,
                        seed), 
                    min_temp(min_temp) {};
    };
}